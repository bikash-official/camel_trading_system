"""
agents/analysis_agent.py
─────────────────────────────────────────────────────────────────
Technical Analysis Agent.

Computes the following indicators on every OHLCV DataFrame:
  • SMA-20  (Simple Moving Average)
  • EMA-12 / EMA-26  (Exponential Moving Averages)
  • MACD + Signal Line
  • RSI-14  (Relative Strength Index)
  • Stochastic %K / %D
  • Bollinger Bands (upper / middle / lower)
  • ATR-14  (Average True Range — volatility proxy)

A CAMEL ChatAgent is used to produce a human-readable market
commentary for each ticker that gets included in the final report.
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

logger = logging.getLogger("AnalysisAgent")


def _safe(series: pd.Series) -> float | None:
    """Return the last non-NaN value of a series, else None."""
    val = series.dropna()
    return float(val.iloc[-1]) if len(val) > 0 else None


class TechnicalAnalysisAgent:
    """
    Runs technical analysis on each OHLCV DataFrame in state.raw_data
    and writes results to state.analyzed_data.
    """

    def __init__(self, use_camel_commentary: bool = False):
        """
        Args:
            use_camel_commentary: If True, a CAMEL ChatAgent writes a
                                  short market commentary per ticker.
                                  Requires a valid API key.
        """
        self.use_camel_commentary = use_camel_commentary
        self._camel_agent = self._build_agent() if use_camel_commentary else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[AnalysisAgent] Analysing {len(state.raw_data)} assets …")
        analyzed: dict = {}

        for ticker, df in state.raw_data.items():
            try:
                result = self._analyse(ticker, df)
                if self._camel_agent:
                    commentary, signal = self._get_commentary(ticker, result)
                    result["commentary"] = commentary
                    if signal:
                        result["signal"] = signal
                analyzed[ticker] = result
            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")
                state.errors.append(f"AnalysisAgent/{ticker}: {e}")

        state.analyzed_data = analyzed
        logger.info(f"[AnalysisAgent] Done. {len(analyzed)} assets analysed.\n")
        return state

    # ── Core indicator calculations ───────────────────────────────────────────
    def _analyse(self, ticker: str, df: pd.DataFrame) -> dict:
        # Ensure we have a proper numeric Close column
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        high  = pd.to_numeric(df["High"],  errors="coerce").dropna()
        low   = pd.to_numeric(df["Low"],   errors="coerce").dropna()

        result: dict = {"ticker": ticker, "current_price": _safe(close)}

        # ── SMA ──────────────────────────────────────────────────────────────
        result["sma_20"]  = _safe(close.rolling(20).mean())
        result["sma_50"]  = _safe(close.rolling(50).mean())

        # ── EMA ──────────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        result["ema_12"] = _safe(ema12)
        result["ema_26"] = _safe(ema26)

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        result["macd"]        = _safe(macd_line)
        result["macd_signal"] = _safe(signal_line)
        result["macd_hist"]   = _safe(macd_line - signal_line)

        # ── RSI-14 ───────────────────────────────────────────────────────────
        result["rsi_14"] = self._rsi(close, 14)

        # ── Stochastic %K / %D ───────────────────────────────────────────────
        k, d = self._stochastic(high, low, close, 14, 3)
        result["stoch_k"] = k
        result["stoch_d"] = d

        # ── Bollinger Bands ───────────────────────────────────────────────────
        sma20    = close.rolling(20).mean()
        std20    = close.rolling(20).std()
        result["bb_upper"]  = _safe(sma20 + 2 * std20)
        result["bb_middle"] = _safe(sma20)
        result["bb_lower"]  = _safe(sma20 - 2 * std20)

        # ── ATR-14 ───────────────────────────────────────────────────────────
        result["atr_14"] = self._atr(high, low, close, 14)

        # ── Volume ───────────────────────────────────────────────────────────
        vol = pd.to_numeric(df.get("Volume", pd.Series(dtype=float)), errors="coerce")
        result["volume"]      = _safe(vol)
        result["avg_volume"]  = _safe(vol.rolling(20).mean())

        # ── Signal derivation ─────────────────────────────────────────────────
        result["signal"] = self._derive_signal(result)

        logger.info(
            f"  ✓ {ticker:15s} | price={result['current_price']:.2f} "
            f"RSI={result['rsi_14']:.1f} Signal={result['signal']}"
        )
        return result

    # ── Indicator implementations ─────────────────────────────────────────────
    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float | None:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return _safe(rsi)

    @staticmethod
    def _stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 14, d_period: int = 3
    ) -> tuple[float | None, float | None]:
        lowest_low   = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
        d = k.rolling(d_period).mean()
        return _safe(k), _safe(d)

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float | None:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return _safe(tr.rolling(period).mean())

    @staticmethod
    def _derive_signal(data: dict) -> str:
        """Simple rule-based signal: BUY / SELL / HOLD."""
        price = data.get("current_price")
        sma   = data.get("sma_20")
        rsi   = data.get("rsi_14")
        macd  = data.get("macd")
        sig   = data.get("macd_signal")

        score = 0
        if price and sma:
            score += 1 if price > sma else -1
        if rsi:
            if rsi < 35:   score += 2   # oversold → buy
            elif rsi > 65: score -= 2   # overbought → sell
        if macd is not None and sig is not None:
            score += 1 if macd > sig else -1

        if score >= 2:   return "BUY"
        elif score <= -2: return "SELL"
        else:            return "HOLD"

    # ── CAMEL commentary ──────────────────────────────────────────────────────
    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.3},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Senior Technical Analyst",
                content=(
                    "You are a senior technical analyst. Given indicator values "
                    "for a stock, write a concise 2-sentence market commentary. "
                    "Be factual, professional, and mention the key signal. "
                    "Your response MUST end with exactly: 'Signal: [BUY/SELL/HOLD]'"
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL analyst agent: {e}")
            return None

    def _get_commentary(self, ticker: str, data: dict) -> tuple[str, str | None]:
        prompt = (
            f"Ticker: {ticker}\n"
            f"Price: {data.get('current_price'):.2f}\n"
            f"SMA-20: {data.get('sma_20')}\n"
            f"RSI-14: {data.get('rsi_14')}\n"
            f"MACD: {data.get('macd')}\n"
            f"Raw Signal: {data.get('signal')}\n"
            "Write a 2-sentence market commentary and end with 'Signal: [BUY/SELL/HOLD]'."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            text = response.msgs[0].content.strip()
            
            signal = None
            if "Signal:" in text:
                parts = text.split("Signal:", 1)
                commentary = parts[0].strip()
                signal_word = parts[1].strip().upper()
                if signal_word in ["BUY", "SELL", "HOLD"]:
                     signal = signal_word
                else:
                    # just take the first word
                    signal_word = signal_word.split()[0]
                    if signal_word in ["BUY", "SELL", "HOLD"]:
                        signal = signal_word
                return commentary, signal

            return text, None
        except Exception as e:
            return f"Commentary unavailable: {e}", None
