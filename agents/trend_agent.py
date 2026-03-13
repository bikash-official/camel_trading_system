"""
agents/trend_agent.py
─────────────────────────────────────────────────────────────────
Trend Analysis Agent.

Determines the market trend for each ticker:
  • MA-20 vs MA-50 crossover → UPTREND / DOWNTREND / SIDEWAYS
  • ADX  (Average Directional Index) → trend strength (0-100)
  • Trend duration  (consecutive days in current trend)

A CAMEL ChatAgent can optionally provide trend commentary.
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

logger = logging.getLogger("TrendAgent")


class TrendAgent:
    """
    Analyses trend direction and strength for each ticker.
    Results stored in state.trend_signals as ticker → dict.
    """

    def __init__(self, use_camel_commentary: bool = False):
        self.use_camel_commentary = use_camel_commentary
        self._camel_agent = self._build_agent() if use_camel_commentary else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[TrendAgent] Analysing trends for {len(state.raw_data)} assets …")
        signals: dict = {}

        for ticker, df in state.raw_data.items():
            try:
                result = self._analyse_trend(ticker, df)
                if self._camel_agent:
                    commentary, trend, strength = self._get_commentary(ticker, result)
                    result["commentary"] = commentary
                    if trend:
                        result["trend"] = trend
                    if strength:
                        result["trend_strength"] = strength
                signals[ticker] = result
            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")
                state.errors.append(f"TrendAgent/{ticker}: {e}")

        state.trend_signals = signals
        logger.info(f"[TrendAgent] Done. {len(signals)} assets analysed.\n")
        return state

    # ── Trend analysis ────────────────────────────────────────────────────────
    def _analyse_trend(self, ticker: str, df: pd.DataFrame) -> dict:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        high = pd.to_numeric(df["High"], errors="coerce").dropna()
        low = pd.to_numeric(df["Low"], errors="coerce").dropna()

        result: dict = {"ticker": ticker}

        # ── MA crossover trend ───────────────────────────────────────────────
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()

        ma20_last = float(ma20.iloc[-1]) if len(ma20.dropna()) > 0 else None
        ma50_last = float(ma50.iloc[-1]) if len(ma50.dropna()) > 0 else None

        if ma20_last and ma50_last:
            diff_pct = (ma20_last - ma50_last) / ma50_last * 100
            if diff_pct > 1.0:
                result["trend"] = "UPTREND"
            elif diff_pct < -1.0:
                result["trend"] = "DOWNTREND"
            else:
                result["trend"] = "SIDEWAYS"
            result["ma_spread_pct"] = round(diff_pct, 2)
        else:
            result["trend"] = "UNKNOWN"
            result["ma_spread_pct"] = 0.0

        result["ma_20"] = ma20_last
        result["ma_50"] = ma50_last

        # ── ADX  (Average Directional Index) ─────────────────────────────────
        adx = self._adx(high, low, close, period=14)
        result["adx"] = adx

        if adx is not None:
            if adx > 40:
                result["trend_strength"] = "STRONG"
            elif adx > 25:
                result["trend_strength"] = "MODERATE"
            else:
                result["trend_strength"] = "WEAK"
        else:
            result["trend_strength"] = "UNKNOWN"

        # ── Trend duration ───────────────────────────────────────────────────
        result["trend_days"] = self._trend_duration(ma20, ma50)

        # ── Price position ───────────────────────────────────────────────────
        current_price = float(close.iloc[-1])
        result["price_vs_ma20"] = "ABOVE" if current_price > (ma20_last or 0) else "BELOW"
        result["price_vs_ma50"] = "ABOVE" if current_price > (ma50_last or 0) else "BELOW"

        adx_str = f"{adx:.1f}" if adx else "N/A"
        logger.info(
            f"  ✓ {ticker:15s} | {result['trend']:10s} | "
            f"ADX={adx_str} | strength={result['trend_strength']} | "
            f"{result['trend_days']}d"
        )
        return result

    # ── ADX calculation ───────────────────────────────────────────────────────
    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> float | None:
        if len(close) < period * 3:
            return None

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        # When +DM > -DM, keep +DM; else 0 (and vice versa)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()

        val = adx.dropna()
        return float(val.iloc[-1]) if len(val) > 0 else None

    @staticmethod
    def _trend_duration(ma20: pd.Series, ma50: pd.Series) -> int:
        """Count consecutive days MA20 has been above/below MA50."""
        diff = ma20 - ma50
        diff = diff.dropna()
        if len(diff) == 0:
            return 0

        current_sign = 1 if diff.iloc[-1] > 0 else -1
        count = 0
        for val in reversed(diff.values):
            if (val > 0 and current_sign > 0) or (val < 0 and current_sign < 0):
                count += 1
            else:
                break
        return count

    # ── CAMEL commentary ──────────────────────────────────────────────────────
    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.3},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Trend Analyst",
                content=(
                    "You are a trend analyst. Given trend data for a stock, "
                    "write a concise 2-sentence summary of the current trend "
                    "and what it means for traders. "
                    "Your response MUST end with exactly: 'Trend: [UPTREND/DOWNTREND/SIDEWAYS] | Strength: [STRONG/MODERATE/WEAK]'"
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL trend agent: {e}")
            return None

    def _get_commentary(self, ticker: str, data: dict) -> tuple[str, str | None, str | None]:
        prompt = (
            f"Ticker: {ticker}\n"
            f"Raw Trend: {data.get('trend')}\n"
            f"ADX: {data.get('adx')}\n"
            f"Raw Strength: {data.get('trend_strength')}\n"
            f"Duration: {data.get('trend_days')} days\n"
            "Write a 2-sentence trend commentary and end with 'Trend: [UPTREND/DOWNTREND/SIDEWAYS] | Strength: [STRONG/MODERATE/WEAK]'."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            text = response.msgs[0].content.strip()
            
            trend = None
            strength = None
            
            if "Trend:" in text:
                parts = text.split("Trend:", 1)
                commentary = parts[0].strip()
                
                rest = parts[1]
                if "|" in rest:
                    t_str, s_str = rest.split("|", 1)
                    
                    # Parse Trend
                    t_word = t_str.strip().upper()
                    # Remove any trailing spaces or newlines that might cause match failure
                    t_word = t_word.split()[0] if t_word else ""
                    if t_word in ["UPTREND", "DOWNTREND", "SIDEWAYS"]:
                        trend = t_word
                    elif t_word == "UP":
                        trend = "UPTREND"
                    elif t_word == "DOWN":
                        trend = "DOWNTREND"
                        
                    # Parse Strength
                    if "Strength:" in s_str:
                        s_word = s_str.split("Strength:")[1].strip().upper()
                        s_word = s_word.split()[0] if s_word else ""
                        if s_word in ["STRONG", "MODERATE", "WEAK"]:
                            strength = s_word
                            
                return commentary, trend, strength

            return text, None, None
        except Exception as e:
            return f"Trend commentary unavailable: {e}", None, None
