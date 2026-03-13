"""
agents/data_agent.py
─────────────────────────────────────────────────────────────────
CAMEL-powered Data Collection Agent.

Responsibilities:
  • Downloads OHLCV data from Yahoo Finance for all tickers
  • Falls back to local CSV cache when offline
  • Uses a CAMEL ChatAgent to decide which tickers are worth
    analysing based on volume / liquidity filters
"""

from __future__ import annotations
import os
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState

logger = logging.getLogger("DataAgent")

# ── Default universe  (100 tickers across 9 sectors) ─────────────────────────
DEFAULT_TICKERS = [
    # ── US Tech (20) ─────────────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "NFLX", "AMD", "INTC",
    "CRM", "ORCL", "ADBE", "CSCO", "AVGO",
    "QCOM", "TXN", "NOW", "UBER", "SHOP",
    # ── Finance (10) ─────────────────────────────────────────────────────────
    "JPM", "BAC", "GS", "V", "MA",
    "WFC", "MS", "AXP", "C", "BLK",
    # ── Healthcare (10) ──────────────────────────────────────────────────────
    "JNJ", "PFE", "MRNA", "UNH", "ABBV",
    "LLY", "TMO", "ABT", "BMY", "AMGN",
    # ── Consumer (10) ────────────────────────────────────────────────────────
    "WMT", "PG", "KO", "PEP", "COST",
    "MCD", "NKE", "SBUX", "HD", "LOW",
    # ── Energy (10) ──────────────────────────────────────────────────────────
    "XOM", "CVX", "BP", "COP", "SLB",
    "EOG", "OXY", "MPC", "PSX", "VLO",
    # ── Industrials (10) ─────────────────────────────────────────────────────
    "CAT", "BA", "HON", "UPS", "RTX",
    "GE", "MMM", "LMT", "DE", "UNP",
    # ── Indian Stocks NSE (15) ───────────────────────────────────────────────
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS",
    # ── Crypto (5) ───────────────────────────────────────────────────────────
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    # ── ETFs (10) ────────────────────────────────────────────────────────────
    "SPY", "QQQ", "GLD", "IWM", "DIA",
    "EEM", "VTI", "TLT", "XLF", "XLE",
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _build_camel_agent() -> ChatAgent:
    """Create a lightweight CAMEL ChatAgent for ticker filtering decisions."""
    try:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0.0},
        )
        system_msg = BaseMessage.make_assistant_message(
            role_name="Market Data Analyst",
            content=(
                "You are a market data analyst. When given a list of tickers "
                "and their basic stats, you return a clean JSON list of the "
                "tickers that have sufficient data quality (non-zero volume, "
                "at least 30 rows). Respond ONLY with a JSON array of strings."
            ),
        )
        return ChatAgent(system_message=system_msg, model=model)
    except Exception as e:
        logger.warning(f"Could not build CAMEL agent (no API key?): {e}")
        return None


class DataCollectionAgent:
    """
    Downloads Yahoo Finance OHLCV data for each ticker and stores
    the resulting DataFrames in AgentState.raw_data.
    """

    def __init__(self, use_camel_filter: bool = True):
        self.use_camel_filter = use_camel_filter
        self._camel_agent = _build_camel_agent() if use_camel_filter else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        tickers = state.tickers or DEFAULT_TICKERS
        logger.info(f"[DataAgent] Fetching data for {len(tickers)} tickers …")

        raw: dict = {}
        for ticker in tickers:
            df = self._fetch(ticker, state.period, state.interval)
            if df is not None and not df.empty:
                raw[ticker] = df
                logger.info(f"  ✓ {ticker:15s} → {len(df)} rows")
            else:
                logger.warning(f"  ✗ {ticker:15s} → no data")

        # Optionally use CAMEL agent to filter out low-quality tickers
        if self._camel_agent and raw:
            raw = self._camel_filter(raw)

        state.raw_data = raw
        state.tickers = list(raw.keys())
        logger.info(f"[DataAgent] Done. {len(raw)} assets ready.\n")
        return state

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame | None:
        # 1. Try live download
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self._save_cache(ticker, df)
                return df
        except Exception as e:
            logger.debug(f"Live fetch failed for {ticker}: {e}")

        # 2. Fall back to cached CSV
        return self._load_cache(ticker)

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"
        df.to_csv(path)

    def _load_cache(self, ticker: str) -> pd.DataFrame | None:
        path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"
        if path.exists():
            logger.info(f"  [cache] Loaded {ticker} from {path}")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return None

    def _camel_filter(self, raw: dict) -> dict:
        """Ask CAMEL agent which tickers have good enough data."""
        import json
        summary = {
            t: {"rows": len(df), "avg_volume": int(df["Volume"].mean()) if "Volume" in df.columns else 0}
            for t, df in raw.items()
        }
        prompt = f"Filter these tickers and return only quality ones as a JSON array:\n{json.dumps(summary, indent=2)}"
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            text = response.msgs[0].content.strip()
            # Extract JSON array
            start, end = text.find("["), text.rfind("]") + 1
            good_tickers = json.loads(text[start:end])
            logger.info(f"[CAMEL Filter] Kept {len(good_tickers)}/{len(raw)} tickers")
            return {t: raw[t] for t in good_tickers if t in raw}
        except Exception as e:
            logger.warning(f"CAMEL filter failed, keeping all tickers: {e}")
            return raw
