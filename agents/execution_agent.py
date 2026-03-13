"""
agents/execution_agent.py
─────────────────────────────────────────────────────────────────
Paper Trading Execution Agent.

Takes decisions from state.decisions and simulates trades against
a paper portfolio stored in state.portfolio.

Risk rules enforced:
  • Max 2 % of total portfolio value per trade
  • No short selling
  • No trades with zero/negative prices
  • Skips tickers with HOLD signal
"""

from __future__ import annotations
import logging
from datetime import datetime

from core.state import AgentState

logger = logging.getLogger("ExecutionAgent")

MAX_POSITION_PCT = 0.02   # max 2 % of portfolio per trade
TRANSACTION_COST = 0.001  # 0.1 % per trade


class ExecutionAgent:
    """Executes paper trades and manages the portfolio."""

    def run(self, state: AgentState) -> AgentState:
        portfolio = state.portfolio
        total_value = portfolio["cash"] + sum(
            portfolio["holdings"].get(t, {}).get("shares", 0) *
            (state.analyzed_data.get(t, {}).get("current_price") or 0)
            for t in portfolio["holdings"]
        )
        portfolio["total_value"] = total_value

        logger.info(
            f"[ExecutionAgent] Portfolio: ${total_value:,.2f} | "
            f"Cash: ${portfolio['cash']:,.2f}"
        )

        executed = skipped = 0
        for ticker, action in state.decisions.items():
            price = (state.analyzed_data.get(ticker) or {}).get("current_price")
            if not price or price <= 0:
                logger.warning(f"  ✗ {ticker}: invalid price, skipping")
                skipped += 1
                continue

            if action == "BUY":
                result = self._buy(ticker, price, portfolio, total_value)
            elif action == "SELL":
                result = self._sell(ticker, price, portfolio)
            else:
                skipped += 1
                continue

            if result:
                executed += 1
                portfolio["trades"].append({
                    "time":   datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": action,
                    "price":  price,
                    "shares": result["shares"],
                    "value":  result["value"],
                })

        # Recalculate total portfolio value
        updated_value = portfolio["cash"] + sum(
            portfolio["holdings"].get(t, {}).get("shares", 0) *
            (state.analyzed_data.get(t, {}).get("current_price") or 0)
            for t in portfolio["holdings"]
        )
        portfolio["total_value"] = updated_value
        portfolio["pnl"] = updated_value - 100_000.0   # vs starting cash

        logger.info(
            f"[ExecutionAgent] Executed {executed} trades, "
            f"skipped {skipped}. "
            f"Portfolio value: ${updated_value:,.2f} "
            f"(P&L: ${portfolio['pnl']:+,.2f})\n"
        )
        return state

    # ── Trade helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _buy(ticker: str, price: float, portfolio: dict, total_value: float) -> dict | None:
        max_spend  = total_value * MAX_POSITION_PCT
        spend      = min(max_spend, portfolio["cash"])
        if spend < price:
            logger.warning(f"  ✗ {ticker}: not enough cash to BUY even 1 share")
            return None

        shares = (spend / price) * (1 - TRANSACTION_COST)
        cost   = shares * price * (1 + TRANSACTION_COST)

        portfolio["cash"] -= cost
        holding = portfolio["holdings"].setdefault(ticker, {"shares": 0.0, "avg_price": 0.0})
        total_shares = holding["shares"] + shares
        holding["avg_price"] = (
            (holding["avg_price"] * holding["shares"] + price * shares) / total_shares
        )
        holding["shares"] = total_shares

        logger.info(f"  ✓ BUY  {ticker:12s} {shares:.4f} shares @ ${price:.2f} (cost ${cost:.2f})")
        return {"shares": shares, "value": cost}

    @staticmethod
    def _sell(ticker: str, price: float, portfolio: dict) -> dict | None:
        holding = portfolio["holdings"].get(ticker)
        if not holding or holding["shares"] <= 0:
            logger.info(f"  ✗ SELL {ticker}: no shares held")
            return None

        shares   = holding["shares"]
        proceeds = shares * price * (1 - TRANSACTION_COST)
        realized_pnl = (price - holding["avg_price"]) * shares

        portfolio["cash"] += proceeds
        portfolio["holdings"].pop(ticker)

        logger.info(
            f"  ✓ SELL {ticker:12s} {shares:.4f} shares @ ${price:.2f} "
            f"(proceeds ${proceeds:.2f}, P&L ${realized_pnl:+.2f})"
        )
        return {"shares": shares, "value": proceeds}
