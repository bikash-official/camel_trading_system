"""
agents/portfolio_manager.py
─────────────────────────────────────────────────────────────────
Portfolio Manager Agent.

Comprehensive portfolio tracking and analytics:
  • Running Sharpe ratio
  • Win rate  (% of profitable trades)
  • Max drawdown
  • Position-level P&L
  • Historical portfolio value curve
  • Total return and annualised return

A CAMEL ChatAgent can optionally summarise portfolio performance.
"""

from __future__ import annotations
import logging
import numpy as np
from datetime import datetime
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState

logger = logging.getLogger("PortfolioManager")


class PortfolioManager:
    """
    Tracks comprehensive portfolio metrics.
    Results stored in state.portfolio_metrics.
    """

    def __init__(self, use_camel_summary: bool = False):
        self.use_camel_summary = use_camel_summary
        self._camel_agent = self._build_agent() if use_camel_summary else None
        self._value_history: list[float] = []

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info("[PortfolioManager] Computing portfolio analytics …")
        portfolio = state.portfolio
        initial_cash = 100_000.0

        # Track historical values
        current_value = portfolio.get("total_value", initial_cash)
        self._value_history.append(current_value)

        trades = portfolio.get("trades", [])
        holdings = portfolio.get("holdings", {})

        metrics: dict = {}

        # ── Basic metrics ────────────────────────────────────────────────────
        metrics["total_value"] = current_value
        metrics["cash"] = portfolio.get("cash", initial_cash)
        metrics["total_pnl"] = current_value - initial_cash
        metrics["total_return_pct"] = round(
            ((current_value - initial_cash) / initial_cash) * 100, 2
        )
        metrics["total_trades"] = len(trades)
        metrics["active_positions"] = len(holdings)

        # ── Win rate ─────────────────────────────────────────────────────────
        metrics.update(self._win_rate(trades, state))

        # ── Sharpe ratio ─────────────────────────────────────────────────────
        metrics["sharpe_ratio"] = self._sharpe_ratio(self._value_history)

        # ── Max drawdown ─────────────────────────────────────────────────────
        metrics["max_drawdown"] = self._max_drawdown(self._value_history)
        metrics["max_drawdown_pct"] = f"{metrics['max_drawdown'] * 100:.2f}%"

        # ── Position-level P&L ───────────────────────────────────────────────
        position_pnl = {}
        for ticker, holding in holdings.items():
            price = (state.analyzed_data.get(ticker, {}).get("current_price") or
                     holding.get("avg_price", 0))
            shares = holding.get("shares", 0)
            avg_price = holding.get("avg_price", 0)
            unrealized = (price - avg_price) * shares
            position_pnl[ticker] = {
                "shares": round(shares, 4),
                "avg_price": round(avg_price, 2),
                "current_price": round(price, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pct": round(
                    ((price - avg_price) / avg_price * 100) if avg_price > 0 else 0, 2
                ),
            }
        metrics["positions"] = position_pnl

        # ── Profit factor ────────────────────────────────────────────────────
        metrics["profit_factor"] = self._profit_factor(trades, state)

        # ── Value history ────────────────────────────────────────────────────
        metrics["value_history"] = self._value_history.copy()

        # ── Timestamp ────────────────────────────────────────────────────────
        metrics["last_updated"] = datetime.now().isoformat()

        # ── CAMEL summary ────────────────────────────────────────────────────
        if self._camel_agent:
            metrics["performance_summary"] = self._get_summary(metrics)

        state.portfolio_metrics = metrics

        # ── Log summary ──────────────────────────────────────────────────────
        logger.info(
            f"  Total Value  : ${current_value:>12,.2f}"
        )
        logger.info(
            f"  P&L          : ${metrics['total_pnl']:>+12,.2f}  "
            f"({metrics['total_return_pct']:+.2f}%)"
        )
        logger.info(
            f"  Sharpe Ratio : {metrics['sharpe_ratio']:.3f}"
        )
        logger.info(
            f"  Max Drawdown : {metrics['max_drawdown_pct']}"
        )
        logger.info(
            f"  Win Rate     : {metrics.get('win_rate_pct', 'N/A')}"
        )
        logger.info(f"[PortfolioManager] Done.\n")
        return state

    # ── Analytics helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _win_rate(trades: list[dict], state: AgentState) -> dict:
        """Calculate win rate from completed (sell) trades."""
        if not trades:
            return {"winning_trades": 0, "losing_trades": 0, "win_rate_pct": "N/A"}

        sells = [t for t in trades if t.get("action") == "SELL"]
        if not sells:
            return {"winning_trades": 0, "losing_trades": 0, "win_rate_pct": "N/A"}

        # For each sell, check if it was profitable
        # (simplified: compare sell price to buy price of same ticker)
        buys: dict = {}
        wins = losses = 0
        for trade in trades:
            ticker = trade["ticker"]
            if trade["action"] == "BUY":
                buys[ticker] = trade.get("price", 0)
            elif trade["action"] == "SELL":
                buy_price = buys.get(ticker, trade.get("price", 0))
                if trade.get("price", 0) >= buy_price:
                    wins += 1
                else:
                    losses += 1

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": f"{win_rate:.1f}%",
            "win_rate": round(win_rate, 2),
        }

    @staticmethod
    def _sharpe_ratio(value_history: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from portfolio value history."""
        if len(value_history) < 3:
            return 0.0

        values = np.array(value_history)
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualise (assume daily)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        daily_rf = risk_free_rate / 252

        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
        return round(float(sharpe), 3)

    @staticmethod
    def _max_drawdown(value_history: list[float]) -> float:
        """Calculate maximum drawdown from peak."""
        if len(value_history) < 2:
            return 0.0

        values = np.array(value_history)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        return float(abs(np.min(drawdowns)))

    @staticmethod
    def _profit_factor(trades: list[dict], state: AgentState) -> float:
        """Ratio of gross profits to gross losses."""
        gross_profit = 0.0
        gross_loss = 0.0

        buys: dict = {}
        for trade in trades:
            ticker = trade["ticker"]
            if trade["action"] == "BUY":
                buys[ticker] = trade.get("price", 0)
            elif trade["action"] == "SELL":
                buy_price = buys.get(ticker, trade.get("price", 0))
                pnl = (trade.get("price", 0) - buy_price) * trade.get("shares", 0)
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 2)

    # ── CAMEL summary ─────────────────────────────────────────────────────────
    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.2},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Portfolio Analyst",
                content=(
                    "You are a portfolio performance analyst. Given portfolio "
                    "metrics, write a concise 3-sentence performance summary. "
                    "Mention returns, risk metrics, and any concerns."
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL portfolio agent: {e}")
            return None

    def _get_summary(self, metrics: dict) -> str:
        prompt = (
            f"Total Return: {metrics.get('total_return_pct', 0)}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}\n"
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A')}\n"
            f"Win Rate: {metrics.get('win_rate_pct', 'N/A')}\n"
            f"Profit Factor: {metrics.get('profit_factor', 0)}\n"
            f"Active Positions: {metrics.get('active_positions', 0)}\n"
            "Provide a 3-sentence performance summary."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            return response.msgs[0].content.strip()
        except Exception as e:
            return f"Performance summary unavailable: {e}"
