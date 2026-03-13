"""
agents/pattern_agent.py
─────────────────────────────────────────────────────────────────
Chart Pattern Detection Agent.

Detects classic chart patterns on OHLCV data:
  • Head and Shoulders  (bearish reversal)
  • Double Top / Double Bottom
  • Triangle  (ascending / descending / symmetrical)
  • Bull Flag / Bear Flag

A CAMEL ChatAgent can optionally summarise the detected patterns.
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

logger = logging.getLogger("PatternAgent")


class PatternAgent:
    """
    Scans each ticker's OHLCV DataFrame for chart patterns.
    Results stored in state.pattern_signals as ticker → dict.
    """

    def __init__(self, use_camel_summary: bool = False):
        self.use_camel_summary = use_camel_summary
        self._camel_agent = self._build_agent() if use_camel_summary else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[PatternAgent] Scanning {len(state.raw_data)} assets for chart patterns …")
        signals: dict = {}

        for ticker, df in state.raw_data.items():
            try:
                result = self._detect_patterns(ticker, df)
                if self._camel_agent:
                    summary, bias = self._get_summary(ticker, result)
                    result["summary"] = summary
                    if bias:
                        result["overall_pattern_bias"] = bias
                signals[ticker] = result
            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")
                state.errors.append(f"PatternAgent/{ticker}: {e}")

        state.pattern_signals = signals
        logger.info(f"[PatternAgent] Done. {len(signals)} assets scanned.\n")
        return state

    # ── Pattern detection ────────────────────────────────────────────────────
    def _detect_patterns(self, ticker: str, df: pd.DataFrame) -> dict:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        high = pd.to_numeric(df["High"], errors="coerce").dropna()
        low = pd.to_numeric(df["Low"], errors="coerce").dropna()

        patterns_found = []
        overall = "NEUTRAL"

        # ── Head and Shoulders ───────────────────────────────────────────────
        has = self._head_and_shoulders(close)
        if has:
            patterns_found.append("Head and Shoulders (Bearish)")

        # ── Double Top ───────────────────────────────────────────────────────
        dt = self._double_top(close)
        if dt:
            patterns_found.append("Double Top (Bearish)")

        # ── Double Bottom ────────────────────────────────────────────────────
        db = self._double_bottom(close)
        if db:
            patterns_found.append("Double Bottom (Bullish)")

        # ── Triangle ─────────────────────────────────────────────────────────
        tri = self._triangle(high, low)
        if tri:
            patterns_found.append(f"{tri} Triangle")

        # ── Bull / Bear Flag ─────────────────────────────────────────────────
        flag = self._flag_pattern(close)
        if flag:
            patterns_found.append(flag)

        # ── MA Cross Pattern ─────────────────────────────────────────────────
        ma_pattern = self._ma_cross_pattern(close)
        if ma_pattern:
            patterns_found.append(ma_pattern)

        # Determine overall sentiment from patterns
        bullish_count = sum(1 for p in patterns_found if "Bullish" in p)
        bearish_count = sum(1 for p in patterns_found if "Bearish" in p)

        if bullish_count > bearish_count:
            overall = "BULLISH"
        elif bearish_count > bullish_count:
            overall = "BEARISH"

        result = {
            "ticker": ticker,
            "patterns": patterns_found,
            "pattern_count": len(patterns_found),
            "overall_pattern_bias": overall,
        }
        logger.info(f"  ✓ {ticker:15s} | {len(patterns_found)} patterns | {overall}")
        return result

    # ── Individual pattern detectors ──────────────────────────────────────────
    @staticmethod
    def _head_and_shoulders(close: pd.Series, window: int = 20) -> bool:
        """Detect H&S by finding three peaks where middle is highest."""
        if len(close) < window * 5:
            return False
        recent = close.iloc[-window * 5:]
        peaks = []
        for i in range(window, len(recent) - window):
            if recent.iloc[i] == recent.iloc[i - window:i + window + 1].max():
                peaks.append((i, float(recent.iloc[i])))

        if len(peaks) >= 3:
            # Check if middle peak is highest (head)
            for j in range(1, len(peaks) - 1):
                left, head, right = peaks[j - 1][1], peaks[j][1], peaks[j + 1][1]
                if head > left and head > right:
                    if abs(left - right) / head < 0.05:  # shoulders roughly equal
                        return True
        return False

    @staticmethod
    def _double_top(close: pd.Series, window: int = 15) -> bool:
        """Two peaks at roughly the same level with a valley between."""
        if len(close) < window * 4:
            return False
        recent = close.iloc[-window * 4:]
        peaks = []
        for i in range(window, len(recent) - window):
            if recent.iloc[i] == recent.iloc[i - window:i + window + 1].max():
                peaks.append(float(recent.iloc[i]))
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(p1 - p2) / max(p1, p2) < 0.03:  # within 3%
                return True
        return False

    @staticmethod
    def _double_bottom(close: pd.Series, window: int = 15) -> bool:
        """Two troughs at roughly the same level with a peak between."""
        if len(close) < window * 4:
            return False
        recent = close.iloc[-window * 4:]
        troughs = []
        for i in range(window, len(recent) - window):
            if recent.iloc[i] == recent.iloc[i - window:i + window + 1].min():
                troughs.append(float(recent.iloc[i]))
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(t1 - t2) / max(t1, t2) < 0.03:
                return True
        return False

    @staticmethod
    def _triangle(high: pd.Series, low: pd.Series, lookback: int = 40) -> str | None:
        """Detect converging highs/lows → ascending, descending, or symmetrical triangle."""
        if len(high) < lookback:
            return None
        recent_high = high.iloc[-lookback:]
        recent_low = low.iloc[-lookback:]

        # Linear regression slopes
        x = np.arange(lookback)
        high_slope = np.polyfit(x, recent_high.values, 1)[0]
        low_slope = np.polyfit(x, recent_low.values, 1)[0]

        if high_slope < 0 and low_slope > 0:
            return "Symmetrical"
        elif high_slope < -0.01 and abs(low_slope) < 0.01:
            return "Descending (Bearish)"
        elif low_slope > 0.01 and abs(high_slope) < 0.01:
            return "Ascending (Bullish)"
        return None

    @staticmethod
    def _flag_pattern(close: pd.Series, pole_len: int = 10, flag_len: int = 15) -> str | None:
        """Detect bull/bear flag: strong move followed by consolidation."""
        if len(close) < pole_len + flag_len + 5:
            return None
        pole = close.iloc[-(pole_len + flag_len):-flag_len]
        flag = close.iloc[-flag_len:]

        pole_return = (float(pole.iloc[-1]) - float(pole.iloc[0])) / float(pole.iloc[0])
        flag_std = float(flag.std()) / float(flag.mean())

        if pole_return > 0.05 and flag_std < 0.02:
            return "Bull Flag (Bullish)"
        elif pole_return < -0.05 and flag_std < 0.02:
            return "Bear Flag (Bearish)"
        return None

    @staticmethod
    def _ma_cross_pattern(close: pd.Series) -> str | None:
        """Golden cross / death cross detection."""
        if len(close) < 55:
            return None
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-5] < ma50.iloc[-5]:
            return "Golden Cross (Bullish)"
        elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-5] > ma50.iloc[-5]:
            return "Death Cross (Bearish)"
        return None

    # ── CAMEL summary ────────────────────────────────────────────────────────
    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.2},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Chart Pattern Analyst",
                content=(
                    "You are a chart pattern analyst. Given detected patterns "
                    "for a stock, write a concise 2-sentence summary of what "
                    "these patterns suggest for near-term price action. "
                    "Your response MUST end with exactly: 'Bias: [BULLISH/BEARISH/NEUTRAL]'"
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL pattern agent: {e}")
            return None

    def _get_summary(self, ticker: str, data: dict) -> tuple[str, str | None]:
        patterns = ", ".join(data["patterns"]) if data["patterns"] else "No patterns detected"
        prompt = (
            f"Ticker: {ticker}\n"
            f"Patterns detected: {patterns}\n"
            f"Raw bias: {data['overall_pattern_bias']}\n"
            "Write a 2-sentence pattern analysis summary and end with 'Bias: [BULLISH/BEARISH/NEUTRAL]'."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            text = response.msgs[0].content.strip()
            
            bias = None
            if "Bias:" in text:
                parts = text.split("Bias:", 1)
                summary = parts[0].strip()
                bias_word = parts[1].strip().upper()
                if bias_word in ["BULLISH", "BEARISH", "NEUTRAL"]:
                     bias = bias_word
                else:
                    bias_word = bias_word.split()[0]
                    if bias_word in ["BULLISH", "BEARISH", "NEUTRAL"]:
                         bias = bias_word
                return summary, bias

            return text, None
        except Exception as e:
            return f"Pattern summary unavailable: {e}", None
