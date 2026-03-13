"""
backtesting/backtest_engine.py
─────────────────────────────────────────────────────────────────
Historical Backtesting Engine.

Runs the full 10-agent pipeline over historical data windows,
simulating trading across time to evaluate strategy performance.

Metrics computed:
  • Total Return
  • Sharpe Ratio
  • Max Drawdown
  • Win Rate
  • Profit Factor
  • Number of trades
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from core.state import AgentState
from agents.data_agent        import DataCollectionAgent
from agents.analysis_agent    import TechnicalAnalysisAgent
from agents.pattern_agent     import PatternAgent
from agents.trend_agent       import TrendAgent
from agents.sentiment_agent   import SentimentAnalysisAgent
from agents.strategy_agent    import StrategyAgent
from agents.feature_builder   import FeatureBuilderAgent
from agents.rl_agent          import RLTradingAgent
from agents.risk_agent        import RiskAgent
from agents.execution_agent   import ExecutionAgent
from agents.portfolio_manager import PortfolioManager

logger = logging.getLogger("Backtest")

REPORT_DIR = Path("logs")
REPORT_DIR.mkdir(exist_ok=True)


class BacktestEngine:
    """
    Simulates trading over historical data using the full agent pipeline.
    """

    def __init__(
        self,
        tickers: list[str],
        period: str = "2y",
        initial_cash: float = 100_000.0,
    ):
        self.tickers = tickers
        self.period = period
        self.initial_cash = initial_cash

    def run(self) -> dict:
        """Run the backtest and return performance metrics."""
        logger.info(f"\n{'═' * 60}")
        logger.info(f"  BACKTESTING ENGINE")
        logger.info(f"  Tickers   : {self.tickers}")
        logger.info(f"  Period    : {self.period}")
        logger.info(f"  Cash      : ${self.initial_cash:,.0f}")
        logger.info(f"{'═' * 60}\n")

        # ── Build agents ─────────────────────────────────────────────────────
        data_agent      = DataCollectionAgent(use_camel_filter=False)
        analysis_agent  = TechnicalAnalysisAgent(use_camel_commentary=False)
        pattern_agent   = PatternAgent(use_camel_summary=False)
        trend_agent     = TrendAgent(use_camel_commentary=False)
        sentiment_agent = SentimentAnalysisAgent(use_camel_llm=False)
        strategy_agent  = StrategyAgent(use_camel_reasoning=False)
        feature_builder = FeatureBuilderAgent()
        rl_agent        = RLTradingAgent(use_camel_rationale=False)
        risk_agent      = RiskAgent(use_camel_risk=False)
        execution_agent = ExecutionAgent()
        portfolio_mgr   = PortfolioManager(use_camel_summary=False)

        # ── Create state ─────────────────────────────────────────────────────
        state = AgentState(
            tickers=self.tickers,
            period=self.period,
        )
        state.reset_portfolio(cash=self.initial_cash)

        # ── Run full pipeline ────────────────────────────────────────────────
        pipeline = [
            ("Data Collection",     data_agent.run),
            ("Technical Analysis",  analysis_agent.run),
            ("Pattern Detection",   pattern_agent.run),
            ("Trend Analysis",      trend_agent.run),
            ("Sentiment Analysis",  sentiment_agent.run),
            ("Strategy Combination", strategy_agent.run),
            ("Feature Building",    feature_builder.run),
            ("RL Decisions",        rl_agent.run),
            ("Risk Assessment",     risk_agent.run),
            ("Trade Execution",     execution_agent.run),
            ("Portfolio Analytics", portfolio_mgr.run),
        ]

        for step_name, step_fn in pipeline:
            try:
                state = step_fn(state)
                logger.info(f"  ✓ {step_name}")
            except Exception as e:
                logger.error(f"  ✗ {step_name}: {e}")
                state.errors.append(f"Backtest/{step_name}: {e}")

        # ── Compute backtest metrics ─────────────────────────────────────────
        metrics = self._compute_metrics(state)

        # ── Print report ─────────────────────────────────────────────────────
        self._print_report(metrics)

        # ── Save report ──────────────────────────────────────────────────────
        self._save_report(metrics)

        return metrics

    def _compute_metrics(self, state: AgentState) -> dict:
        """Calculate comprehensive backtesting metrics."""
        portfolio = state.portfolio
        pm = state.portfolio_metrics

        total_value = portfolio.get("total_value", self.initial_cash)
        total_pnl = total_value - self.initial_cash
        total_return = (total_pnl / self.initial_cash) * 100
        trades = portfolio.get("trades", [])

        # Win rate
        buys = {}
        wins = losses = 0
        total_profit = total_loss = 0.0
        for trade in trades:
            ticker = trade["ticker"]
            if trade["action"] == "BUY":
                buys[ticker] = trade.get("price", 0)
            elif trade["action"] == "SELL":
                buy_price = buys.get(ticker, trade.get("price", 0))
                pnl = (trade.get("price", 0) - buy_price) * trade.get("shares", 0)
                if pnl >= 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")

        # Sharpe from portfolio metrics
        sharpe = pm.get("sharpe_ratio", 0) if pm else 0

        # Max drawdown from portfolio metrics
        max_dd = pm.get("max_drawdown", 0) if pm else 0

        metrics = {
            "tickers": self.tickers,
            "period": self.period,
            "initial_cash": self.initial_cash,
            "final_value": total_value,
            "total_pnl": total_pnl,
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_trades": len(trades),
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
            "gross_profit": round(total_profit, 2),
            "gross_loss": round(total_loss, 2),
            "assets_analysed": len(state.tickers),
            "signals": {
                "BUY": sum(1 for d in state.decisions.values() if d == "BUY"),
                "SELL": sum(1 for d in state.decisions.values() if d == "SELL"),
                "HOLD": sum(1 for d in state.decisions.values() if d == "HOLD"),
            },
            "errors": state.errors,
            "timestamp": datetime.now().isoformat(),
        }
        return metrics

    @staticmethod
    def _print_report(metrics: dict) -> None:
        """Print a formatted backtest results report."""
        print(f"\n{'═' * 70}")
        print(f"        📊  BACKTEST RESULTS")
        print(f"        {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        print(f"{'═' * 70}")
        print()
        print(f"── CONFIGURATION ────────────────────────────────────────────────")
        print(f"  Tickers         : {', '.join(metrics['tickers'])}")
        print(f"  Period          : {metrics['period']}")
        print(f"  Initial Cash    : ${metrics['initial_cash']:>12,.2f}")
        print()
        print(f"── PERFORMANCE ──────────────────────────────────────────────────")
        print(f"  Final Value     : ${metrics['final_value']:>12,.2f}")
        print(f"  Total P&L       : ${metrics['total_pnl']:>+12,.2f}")
        print(f"  Total Return    : {metrics['total_return_pct']:>+10.2f}%")
        print(f"  Sharpe Ratio    : {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:>10.2f}%")
        print()
        print(f"── TRADING STATISTICS ───────────────────────────────────────────")
        print(f"  Total Trades    : {metrics['total_trades']:>10}")
        print(f"  Winning Trades  : {metrics['winning_trades']:>10}")
        print(f"  Losing Trades   : {metrics['losing_trades']:>10}")
        print(f"  Win Rate        : {metrics['win_rate_pct']:>10.1f}%")
        print(f"  Profit Factor   : {metrics['profit_factor']:>10}")
        print(f"  Gross Profit    : ${metrics['gross_profit']:>12,.2f}")
        print(f"  Gross Loss      : ${metrics['gross_loss']:>12,.2f}")
        print()
        print(f"── SIGNAL DISTRIBUTION ──────────────────────────────────────────")
        sig = metrics['signals']
        print(f"  BUY={sig['BUY']}  SELL={sig['SELL']}  HOLD={sig['HOLD']}")
        print()

        if metrics["errors"]:
            print(f"── ERRORS ───────────────────────────────────────────────────────")
            for err in metrics["errors"]:
                print(f"  ⚠ {err}")
            print()

        print(f"{'═' * 70}")
        print(f"  ⚠  For educational purposes only. Not financial advice.")
        print(f"{'═' * 70}\n")

    @staticmethod
    def _save_report(metrics: dict) -> None:
        """Save backtest results to file."""
        import json
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"backtest_{date_str}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"  Backtest report saved to {path}")
