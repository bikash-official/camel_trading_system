"""
core/state.py
─────────────────────────────────────────────────────────────────
Shared AgentState object that flows through the entire pipeline.
Every agent reads from and writes to this single object so that
all data stays consistent without global variables.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    # ── Market data (set by DataAgent) ───────────────────────────────────────
    tickers: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)   # ticker → DataFrame

    # ── Technical indicators (set by AnalysisAgent) ───────────────────────────
    analyzed_data: dict[str, Any] = field(default_factory=dict)  # ticker → dict

    # ── Pattern signals (set by PatternAgent) ─────────────────────────────────
    pattern_signals: dict[str, Any] = field(default_factory=dict)  # ticker → dict

    # ── Trend signals (set by TrendAgent) ─────────────────────────────────────
    trend_signals: dict[str, Any] = field(default_factory=dict)  # ticker → dict

    # ── Sentiment scores (set by SentimentAgent) ──────────────────────────────
    sentiment_scores: dict[str, float] = field(default_factory=dict)

    # ── Strategy signals (set by StrategyAgent) ───────────────────────────────
    strategy_signals: dict[str, Any] = field(default_factory=dict)  # ticker → dict

    # ── RL feature vectors (set by FeatureBuilderAgent) ───────────────────────
    feature_vectors: dict[str, list[float]] = field(default_factory=dict)

    # ── RL decisions (set by RLAgent) ─────────────────────────────────────────
    decisions: dict[str, str] = field(default_factory=dict)     # ticker → BUY/SELL/HOLD
    confidence: dict[str, float] = field(default_factory=dict)  # ticker → probability

    # ── Risk assessments (set by RiskAgent) ───────────────────────────────────
    risk_assessments: dict[str, Any] = field(default_factory=dict)

    # ── Portfolio / execution (set by ExecutionAgent) ─────────────────────────
    portfolio: dict[str, Any] = field(default_factory=lambda: {
        "cash": 100_000.0,
        "holdings": {},
        "trades": [],
        "total_value": 100_000.0,
        "pnl": 0.0,
    })

    # ── Portfolio metrics (set by PortfolioManager) ───────────────────────────
    portfolio_metrics: dict[str, Any] = field(default_factory=dict)

    # ── Errors & metadata ────────────────────────────────────────────────────
    errors: list[str] = field(default_factory=list)
    period: str = "6mo"
    interval: str = "1d"

    def reset_portfolio(self, cash: float = 100_000.0) -> None:
        self.portfolio = {
            "cash": cash,
            "holdings": {},
            "trades": [],
            "total_value": cash,
            "pnl": 0.0,
        }
