"""
core/workflow.py
─────────────────────────────────────────────────────────────────
CAMEL-powered Multi-Agent Workflow Orchestrator.

This module wires together ALL 10 agents into a sequential pipeline
and also exposes a CAMEL role-play "supervisor" that narrates
what is happening at each step — demonstrating proper CAMEL usage.

Pipeline steps  (10-agent architecture)
──────────────────────────────────────────
   1  DataCollectionAgent      ← Download OHLCV from Yahoo Finance
   2  TechnicalAnalysisAgent   ← SMA, RSI, MACD, Bollinger, Stochastic
   3  PatternAgent             ← Chart pattern detection
   4  TrendAgent               ← Trend direction & strength
   5  SentimentAnalysisAgent   ← Keyword / LLM news scoring
   6  StrategyAgent            ← Combine all analysis into unified signal
   7  FeatureBuilderAgent      ← Build 7-element RL observation vectors
   8  RLTradingAgent           ← PPO model predictions
   9  RiskAgent                ← Portfolio risk assessment & veto
  10  ExecutionAgent + PortfolioManager + ReportAgent
"""

from __future__ import annotations
import logging
import time
from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState
from agents.data_agent         import DataCollectionAgent
from agents.analysis_agent     import TechnicalAnalysisAgent
from agents.pattern_agent      import PatternAgent
from agents.trend_agent        import TrendAgent
from agents.sentiment_agent    import SentimentAnalysisAgent
from agents.strategy_agent     import StrategyAgent
from agents.feature_builder    import FeatureBuilderAgent
from agents.rl_agent           import RLTradingAgent
from agents.risk_agent         import RiskAgent
from agents.execution_agent    import ExecutionAgent
from agents.portfolio_manager  import PortfolioManager
from agents.report_agent       import ReportAgent

logger = logging.getLogger("Workflow")


def _build_supervisor() -> Optional[ChatAgent]:
    """
    Optional CAMEL supervisor agent that produces a short narrative
    commentary at each pipeline step.
    """
    try:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0.4},
        )
        sys_msg = BaseMessage.make_assistant_message(
            role_name="Trading System Supervisor",
            content=(
                "You supervise a multi-agent trading pipeline with 10 agents. "
                "When given a step name and brief status, write ONE concise "
                "sentence describing what just happened and what to expect next."
            ),
        )
        return ChatAgent(system_message=sys_msg, model=model)
    except Exception:
        return None


class TradingWorkflow:
    """
    Orchestrates all 10 agents in sequence. Pass in an AgentState to control
    tickers, period, initial cash, and which optional features to enable.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        period: str = "6mo",
        initial_cash: float = 100_000.0,
        use_camel_commentary: bool = False,  # enables LLM commentary (needs API key)
        use_camel_sentiment:  bool = False,  # enables LLM sentiment  (needs API key)
        use_camel_rationale:  bool = False,  # enables LLM rationale  (needs API key)
        use_supervisor:       bool = False,  # enables pipeline supervisor (needs API key)
    ):
        camel_all = use_camel_commentary  # shorthand

        # ── Agents ───────────────────────────────────────────────────────────
        self.data_agent        = DataCollectionAgent(use_camel_filter=camel_all)
        self.analysis_agent    = TechnicalAnalysisAgent(use_camel_commentary=use_camel_commentary)
        self.pattern_agent     = PatternAgent(use_camel_summary=camel_all)
        self.trend_agent       = TrendAgent(use_camel_commentary=camel_all)
        self.sentiment_agent   = SentimentAnalysisAgent(use_camel_llm=use_camel_sentiment)
        self.strategy_agent    = StrategyAgent(use_camel_reasoning=camel_all)
        self.feature_builder   = FeatureBuilderAgent()
        self.rl_agent          = RLTradingAgent(use_camel_rationale=use_camel_rationale)
        self.risk_agent        = RiskAgent(use_camel_risk=camel_all)
        self.execution_agent   = ExecutionAgent()
        self.portfolio_manager = PortfolioManager(use_camel_summary=camel_all)
        self.report_agent      = ReportAgent()
        self.supervisor        = _build_supervisor() if use_supervisor else None

        # ── Initial state ────────────────────────────────────────────────────
        self.initial_state = AgentState(
            tickers=tickers or [],
            period=period,
        )
        self.initial_state.reset_portfolio(cash=initial_cash)

    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> AgentState:
        state = self.initial_state
        self._banner("MULTI-AGENT TRADING SYSTEM STARTING  (10-Agent Pipeline)")

        steps = [
            ("1/10  Data Collection",       self.data_agent.run),
            ("2/10  Technical Analysis",     self.analysis_agent.run),
            ("3/10  Pattern Detection",      self.pattern_agent.run),
            ("4/10  Trend Analysis",         self.trend_agent.run),
            ("5/10  Sentiment Analysis",     self.sentiment_agent.run),
            ("6/10  Strategy Combination",   self.strategy_agent.run),
            ("7/10  Feature Building",       self.feature_builder.run),
            ("8/10  RL Decision Making",     self.rl_agent.run),
            ("9/10  Risk Assessment",        self.risk_agent.run),
            ("10/10 Trade Execution",        self._execute_and_report),
        ]

        for step_name, step_fn in steps:
            logger.info(f"{'─'*60}")
            logger.info(f"  STEP {step_name}")
            logger.info(f"{'─'*60}")
            t0 = time.time()

            try:
                state = step_fn(state)
            except Exception as e:
                logger.error(f"  ✗ Step failed: {e}")
                state.errors.append(f"{step_name}: {e}")
                continue

            elapsed = time.time() - t0
            logger.info(f"  ✓ Completed in {elapsed:.2f}s")

            # Supervisor commentary (requires API key)
            if self.supervisor:
                self._supervise(step_name, state)

        self._banner("PIPELINE COMPLETE  (10 agents executed)")
        return state

    # ── Combined execution step ───────────────────────────────────────────────
    def _execute_and_report(self, state: AgentState) -> AgentState:
        """Execute trades, update portfolio metrics, and generate report."""
        state = self.execution_agent.run(state)
        state = self.portfolio_manager.run(state)
        state = self.report_agent.run(state)
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _supervise(self, step: str, state: AgentState) -> None:
        status = (
            f"Step '{step}' finished. "
            f"Tickers ready: {len(state.tickers)}. "
            f"Decisions made: {len(state.decisions)}. "
            f"Portfolio value: ${state.portfolio.get('total_value', 0):,.0f}."
        )
        try:
            msg = BaseMessage.make_user_message(role_name="User", content=status)
            resp = self.supervisor.step(msg)
            logger.info(f"  🤖 Supervisor: {resp.msgs[0].content.strip()}")
        except Exception:
            pass

    @staticmethod
    def _banner(text: str) -> None:
        line = "═" * 64
        logger.info(f"\n{line}\n  {text}\n{line}")
