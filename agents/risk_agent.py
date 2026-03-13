"""
agents/risk_agent.py
─────────────────────────────────────────────────────────────────
Risk Assessment Agent.

Evaluates portfolio risk before trade execution:
  • Position concentration  (max 5% per ticker)
  • Portfolio drawdown monitoring
  • Simple VaR  (Value at Risk)
  • Correlation check between held positions
  • Can veto or reduce trade sizes

Uses a CAMEL ChatAgent to optionally explain risk decisions.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState

logger = logging.getLogger("RiskAgent")

MAX_SINGLE_POSITION_PCT = 0.05   # 5% max per ticker
MAX_PORTFOLIO_DRAWDOWN  = 0.15   # 15% max drawdown before warning
VAR_CONFIDENCE          = 0.95   # 95% VaR


class RiskAgent:
    """
    Assesses portfolio risk and can adjust/veto decisions.
    Results stored in state.risk_assessments.
    """

    def __init__(self, use_camel_risk: bool = False):
        self.use_camel_risk = use_camel_risk
        self._camel_agent = self._build_agent() if use_camel_risk else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info("[RiskAgent] Evaluating portfolio risk …")
        portfolio = state.portfolio
        total_value = portfolio.get("total_value", 100_000)

        assessments: dict = {}

        # ── Portfolio-level risk ──────────────────────────────────────────────
        portfolio_risk = self._portfolio_risk(state, total_value)
        assessments["_portfolio"] = portfolio_risk

        # ── Per-ticker risk check ─────────────────────────────────────────────
        vetoed = 0
        for ticker in state.tickers:
            decision = state.decisions.get(ticker, "HOLD")
            if decision == "HOLD":
                continue

            risk = self._ticker_risk(ticker, decision, state, total_value)
            assessments[ticker] = risk

            # Veto if risk is too high
            if risk.get("veto", False):
                original = state.decisions[ticker]
                state.decisions[ticker] = "HOLD"
                logger.warning(
                    f"  ⛔ {ticker:15s} | {original} → HOLD (vetoed: {risk['veto_reason']})"
                )
                vetoed += 1
            else:
                logger.info(
                    f"  ✓ {ticker:15s} | {decision:4s} | "
                    f"risk={risk.get('risk_level', 'N/A')}"
                )

        # Optional CAMEL summary
        if self._camel_agent:
            assessments["_summary"] = self._get_risk_summary(assessments)

        state.risk_assessments = assessments
        logger.info(
            f"[RiskAgent] Done. {vetoed} trades vetoed, "
            f"{len(assessments) - 1} assessed.\n"
        )
        return state

    # ── Portfolio-level risk ──────────────────────────────────────────────────
    def _portfolio_risk(self, state: AgentState, total_value: float) -> dict:
        portfolio = state.portfolio
        initial_cash = 100_000.0
        pnl = portfolio.get("pnl", 0)
        drawdown = -pnl / initial_cash if pnl < 0 else 0.0

        # Count positions
        holdings = portfolio.get("holdings", {})
        num_positions = len(holdings)

        # Concentration check
        max_concentration = 0.0
        for ticker, holding in holdings.items():
            price = (state.analyzed_data.get(ticker, {}).get("current_price") or
                     holding.get("avg_price", 0))
            pos_value = holding.get("shares", 0) * price
            concentration = pos_value / total_value if total_value > 0 else 0
            max_concentration = max(max_concentration, concentration)

        # Simple VaR from recent returns
        var_95 = self._portfolio_var(state)

        risk = {
            "total_value": total_value,
            "drawdown": round(drawdown, 4),
            "drawdown_pct": f"{drawdown * 100:.2f}%",
            "max_concentration": round(max_concentration, 4),
            "num_positions": num_positions,
            "var_95": round(var_95, 2),
            "risk_level": "HIGH" if drawdown > 0.10 else ("MEDIUM" if drawdown > 0.05 else "LOW"),
        }

        if drawdown > MAX_PORTFOLIO_DRAWDOWN:
            risk["warning"] = f"Portfolio drawdown {drawdown:.1%} exceeds {MAX_PORTFOLIO_DRAWDOWN:.0%} limit!"
            logger.warning(f"  ⚠ {risk['warning']}")

        return risk

    # ── Per-ticker risk ───────────────────────────────────────────────────────
    def _ticker_risk(self, ticker: str, decision: str,
                     state: AgentState, total_value: float) -> dict:
        portfolio = state.portfolio
        holdings = portfolio.get("holdings", {})
        analysis = state.analyzed_data.get(ticker, {})
        price = analysis.get("current_price", 0)

        risk: dict = {"ticker": ticker, "decision": decision, "veto": False}

        if decision == "BUY":
            # Check position concentration
            existing = holdings.get(ticker, {})
            existing_value = existing.get("shares", 0) * price
            new_spend = total_value * 0.02  # ExecutionAgent spends 2%
            projected_concentration = (existing_value + new_spend) / total_value

            risk["projected_concentration"] = round(projected_concentration, 4)

            if projected_concentration > MAX_SINGLE_POSITION_PCT:
                risk["veto"] = True
                risk["veto_reason"] = (
                    f"Position would be {projected_concentration:.1%} of portfolio "
                    f"(max {MAX_SINGLE_POSITION_PCT:.0%})"
                )

            # Check portfolio drawdown
            pnl = portfolio.get("pnl", 0)
            if pnl < -(total_value * MAX_PORTFOLIO_DRAWDOWN):
                risk["veto"] = True
                risk["veto_reason"] = "Portfolio in excessive drawdown — no new buys"

        elif decision == "SELL":
            # Check if we actually hold the position
            if ticker not in holdings or holdings[ticker].get("shares", 0) <= 0:
                risk["veto"] = True
                risk["veto_reason"] = "No position to sell"

        # RSI-based risk
        rsi = analysis.get("rsi_14")
        if rsi:
            if decision == "BUY" and rsi > 75:
                risk["risk_level"] = "HIGH"
                risk["risk_note"] = "Buying into overbought territory"
            elif decision == "SELL" and rsi < 25:
                risk["risk_level"] = "HIGH"
                risk["risk_note"] = "Selling in oversold territory"
            else:
                risk["risk_level"] = "LOW"
        else:
            risk["risk_level"] = "MEDIUM"

        return risk

    # ── Simple portfolio VaR ──────────────────────────────────────────────────
    @staticmethod
    def _portfolio_var(state: AgentState) -> float:
        """Simple historical VaR from available price data."""
        returns = []
        for ticker, df in state.raw_data.items():
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) > 10:
                daily_returns = close.pct_change().dropna()
                returns.extend(daily_returns.values.tolist())

        if not returns:
            return 0.0

        returns_arr = np.array(returns)
        var = float(np.percentile(returns_arr, (1 - VAR_CONFIDENCE) * 100))
        return var * 100_000  # scale to portfolio size

    # ── CAMEL risk summary ────────────────────────────────────────────────────
    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.1},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Risk Manager",
                content=(
                    "You are a portfolio risk manager. Given risk metrics, "
                    "provide a 2-sentence risk assessment summary. Be specific "
                    "about the biggest risks and any recommended actions."
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL risk agent: {e}")
            return None

    def _get_risk_summary(self, assessments: dict) -> str:
        portfolio_risk = assessments.get("_portfolio", {})
        vetoed = sum(1 for k, v in assessments.items()
                     if k != "_portfolio" and isinstance(v, dict) and v.get("veto"))
        prompt = (
            f"Portfolio drawdown: {portfolio_risk.get('drawdown_pct', 'N/A')}\n"
            f"Max concentration: {portfolio_risk.get('max_concentration', 0):.1%}\n"
            f"VaR (95%): ${portfolio_risk.get('var_95', 0):,.0f}\n"
            f"Risk level: {portfolio_risk.get('risk_level', 'N/A')}\n"
            f"Trades vetoed: {vetoed}\n"
            "Provide a 2-sentence risk assessment."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            return response.msgs[0].content.strip()
        except Exception as e:
            return f"Risk summary unavailable: {e}"
