"""
agents/report_agent.py
─────────────────────────────────────────────────────────────────
Communication / Report Generation Agent.

Generates a comprehensive final report covering:
  • Market overview  (bullish / bearish counts)
  • Signal distribution  (BUY / SELL / HOLD)
  • Pattern & trend analysis summary
  • Strategy agent signals
  • Risk assessment summary
  • Top buy and sell signals ranked by confidence
  • Portfolio summary  (cash, holdings, P&L)
  • Portfolio analytics  (Sharpe, drawdown, win rate)
  • Any errors that occurred during the pipeline

Optionally uses a CAMEL ChatAgent to produce an executive summary.
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

from core.state import AgentState

logger = logging.getLogger("ReportAgent")
REPORT_DIR = Path("logs")
REPORT_DIR.mkdir(exist_ok=True)


class ReportAgent:
    """Generates and prints the final trading report."""

    def run(self, state: AgentState) -> AgentState:
        report_lines = self._build_report(state)
        report_text  = "\n".join(report_lines)

        # Print to console
        print(report_text)

        # Save to file
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"report_{date_str}.txt"
        path.write_text(report_text, encoding="utf-8")
        logger.info(f"[ReportAgent] Report saved to {path}")

        return state

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_report(state: AgentState) -> list[str]:
        decisions  = state.decisions
        confidence = state.confidence
        analysis   = state.analyzed_data
        portfolio  = state.portfolio

        # Signal counts
        buy_tickers  = [t for t, d in decisions.items() if d == "BUY"]
        sell_tickers = [t for t, d in decisions.items() if d == "SELL"]
        hold_tickers = [t for t, d in decisions.items() if d == "HOLD"]

        # Market mood
        bullish = sum(1 for t in analysis if analysis[t].get("signal") == "BUY")
        bearish = sum(1 for t in analysis if analysis[t].get("signal") == "SELL")
        mood    = "Bullish 📈" if bullish > bearish else ("Bearish 📉" if bearish > bullish else "Neutral ➡️")

        lines = [
            "",
            "═" * 70,
            "        🤖  CAMEL MULTI-AGENT TRADING SYSTEM  —  REPORT",
            "        10-Agent Pipeline  •  CAMEL AI + PPO Reinforcement Learning",
            f"        {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            "═" * 70,
            "",
            "── SUMMARY ──────────────────────────────────────────────────────",
            f"  Assets Analysed   : {len(decisions)}",
            f"  Signal Distribution: BUY={len(buy_tickers)}  SELL={len(sell_tickers)}  HOLD={len(hold_tickers)}",
            f"  Trades Executed   : {len(portfolio.get('trades', []))}",
            "",
            "── MARKET OVERVIEW ──────────────────────────────────────────────",
            f"  Overall Mood : {mood}",
            f"  Bullish       : {bullish}",
            f"  Bearish       : {bearish}",
            f"  Neutral       : {len(analysis) - bullish - bearish}",
            "",
        ]

        # ── Pattern & Trend Summary ──────────────────────────────────────────
        if state.pattern_signals:
            lines.append("── PATTERN ANALYSIS ─────────────────────────────────────────────")
            for t in list(state.pattern_signals.keys())[:10]:
                ps = state.pattern_signals[t]
                patterns = ", ".join(ps.get("patterns", [])) or "None"
                lines.append(f"  {t:15s} | {ps.get('overall_pattern_bias', 'N/A'):8s} | {patterns}")
            lines.append("")

        if state.trend_signals:
            lines.append("── TREND ANALYSIS ───────────────────────────────────────────────")
            for t in list(state.trend_signals.keys())[:10]:
                ts = state.trend_signals[t]
                adx_val = ts.get('adx')
                adx_str = f"{adx_val:.1f}" if adx_val else "N/A"
                lines.append(
                    f"  {t:15s} | {ts.get('trend', 'N/A'):10s} | "
                    f"ADX={adx_str:>5} | "
                    f"{ts.get('trend_strength', 'N/A')} | {ts.get('trend_days', 0)}d"
                )
            lines.append("")

        # ── Strategy Signals ─────────────────────────────────────────────────
        if state.strategy_signals:
            lines.append("── STRATEGY SIGNALS ─────────────────────────────────────────────")
            for t in list(state.strategy_signals.keys())[:10]:
                ss = state.strategy_signals[t]
                lines.append(
                    f"  {t:15s} | {ss.get('signal', 'N/A'):12s} | "
                    f"conf={ss.get('confidence', 0):.2f} | "
                    f"score={ss.get('composite_score', 0):+.3f}"
                )
            lines.append("")

        # ── Risk Assessment ──────────────────────────────────────────────────
        if state.risk_assessments:
            pr = state.risk_assessments.get("_portfolio", {})
            lines += [
                "── RISK ASSESSMENT ──────────────────────────────────────────────",
                f"  Portfolio Risk   : {pr.get('risk_level', 'N/A')}",
                f"  Drawdown         : {pr.get('drawdown_pct', 'N/A')}",
                f"  Max Concentration: {pr.get('max_concentration', 0):.1%}",
                f"  VaR (95%)        : ${pr.get('var_95', 0):,.0f}",
                "",
            ]

        # ── Top Buy Signals ──────────────────────────────────────────────────
        lines.append("── TOP BUY SIGNALS ──────────────────────────────────────────────")
        sorted_buys = sorted(buy_tickers, key=lambda t: confidence.get(t, 0), reverse=True)
        for t in sorted_buys[:8]:
            price = analysis.get(t, {}).get("current_price") or 0
            conf  = confidence.get(t, 0)
            rsi   = analysis.get(t, {}).get("rsi_14") or 0
            lines.append(
                f"  {t:15s} | BUY  | conf={conf:.2f} | ${price:>9.2f} | RSI={rsi:.1f}"
            )

        lines += [
            "",
            "── TOP SELL SIGNALS ─────────────────────────────────────────────",
        ]
        sorted_sells = sorted(sell_tickers, key=lambda t: confidence.get(t, 0), reverse=True)
        for t in sorted_sells[:5]:
            price = analysis.get(t, {}).get("current_price") or 0
            conf  = confidence.get(t, 0)
            lines.append(
                f"  {t:15s} | SELL | conf={conf:.2f} | ${price:>9.2f}"
            )

        # ── Portfolio ────────────────────────────────────────────────────────
        trades = portfolio.get("trades", [])
        pnl    = portfolio.get("pnl", 0)
        pnl_pct = (pnl / 100_000) * 100 if pnl else 0
        lines += [
            "",
            "── PORTFOLIO ────────────────────────────────────────────────────",
            f"  Cash           : ${portfolio.get('cash', 0):>12,.2f}",
            f"  Total Value    : ${portfolio.get('total_value', 0):>12,.2f}",
            f"  Total P&L      : ${pnl:>+12,.2f}  ({pnl_pct:+.2f}%)",
            f"  Total Trades   : {len(trades)}",
            "",
        ]

        # ── Portfolio Analytics ──────────────────────────────────────────────
        pm = state.portfolio_metrics
        if pm:
            lines += [
                "── PORTFOLIO ANALYTICS ──────────────────────────────────────────",
                f"  Sharpe Ratio    : {pm.get('sharpe_ratio', 'N/A')}",
                f"  Max Drawdown    : {pm.get('max_drawdown_pct', 'N/A')}",
                f"  Win Rate        : {pm.get('win_rate_pct', 'N/A')}",
                f"  Profit Factor   : {pm.get('profit_factor', 'N/A')}",
                "",
            ]

        # ── Current Holdings ─────────────────────────────────────────────────
        if portfolio.get("holdings"):
            lines.append("── CURRENT HOLDINGS ──────────────────────────────────────────")
            for ticker, h in portfolio["holdings"].items():
                cur_price = analysis.get(ticker, {}).get("current_price") or h["avg_price"]
                pos_value = h["shares"] * cur_price
                unrealized = (cur_price - h["avg_price"]) * h["shares"]
                lines.append(
                    f"  {ticker:15s} | {h['shares']:.4f} shares "
                    f"@ avg ${h['avg_price']:.2f} | "
                    f"value ${pos_value:,.2f} | "
                    f"unrealized P&L ${unrealized:+,.2f}"
                )
            lines.append("")

        # ── Agents Used ──────────────────────────────────────────────────────
        lines += [
            "── AGENTS USED ──────────────────────────────────────────────────",
            "   1. DataCollectionAgent      (CAMEL ChatAgent)",
            "   2. TechnicalAnalysisAgent   (CAMEL ChatAgent)",
            "   3. PatternAgent             (CAMEL ChatAgent)",
            "   4. TrendAgent               (CAMEL ChatAgent)",
            "   5. SentimentAnalysisAgent   (CAMEL ChatAgent)",
            "   6. StrategyAgent            (CAMEL ChatAgent)",
            "   7. FeatureBuilderAgent",
            "   8. RLTradingAgent           (PPO + CAMEL ChatAgent)",
            "   9. RiskAgent                (CAMEL ChatAgent)",
            "  10. ExecutionAgent + PortfolioManager + ReportAgent",
            "",
        ]

        if state.errors:
            lines += ["── ERRORS ───────────────────────────────────────────────────────"]
            for err in state.errors:
                lines.append(f"  ⚠ {err}")
            lines.append("")

        lines += ["═" * 70, "  ⚠  For educational purposes only. Not financial advice.", "═" * 70, ""]
        return lines
