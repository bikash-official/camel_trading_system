"""
agents/sentiment_agent.py
─────────────────────────────────────────────────────────────────
Sentiment Analysis Agent.

Two modes:
  1. Keyword-based  — fast, no API key required.
  2. CAMEL LLM-based — uses a ChatAgent with GPT to score sentences
                       from yfinance news headlines (if available).

The final sentiment score is a float in [-1.0, +1.0] per ticker.
"""

from __future__ import annotations
import logging
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import yfinance as yf

from core.state import AgentState

logger = logging.getLogger("SentimentAgent")

# ── Keyword dictionaries ──────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "record", "bullish", "upgrade",
    "outperform", "growth", "profit", "revenue", "rally", "soar", "strong",
    "positive", "gain", "gains", "buy", "exceed", "exceeded", "dividend",
    "partnership", "innovation", "breakthrough", "expansion", "upside",
}
NEGATIVE_WORDS = {
    "miss", "misses", "decline", "declines", "bearish", "downgrade",
    "underperform", "loss", "losses", "sell", "crash", "fall", "weak",
    "negative", "risk", "lawsuit", "regulation", "recall", "cut", "cuts",
    "layoff", "layoffs", "bankruptcy", "fraud", "investigation", "warning",
}


class SentimentAnalysisAgent:
    """
    Scores news sentiment for every ticker in state.raw_data.
    Results stored in state.sentiment_scores as floats in [-1, +1].
    """

    def __init__(self, use_camel_llm: bool = False):
        self.use_camel_llm = use_camel_llm
        self._camel_agent = self._build_agent() if use_camel_llm else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[SentimentAgent] Scoring sentiment for {len(state.tickers)} tickers …")
        scores: dict = {}

        for ticker in state.tickers:
            headlines = self._fetch_headlines(ticker)
            if self._camel_agent and headlines:
                score = self._llm_score(ticker, headlines)
            else:
                score = self._keyword_score(headlines)

            scores[ticker] = round(score, 4)
            label = "😊 Positive" if score > 0.1 else ("😟 Negative" if score < -0.1 else "😐 Neutral")
            logger.info(f"  {ticker:15s} → {score:+.3f}  {label}  ({len(headlines)} headlines)")

        state.sentiment_scores = scores
        logger.info(f"[SentimentAgent] Done.\n")
        return state

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fetch_headlines(ticker: str) -> list[str]:
        """Try yfinance news; return list of headline strings."""
        try:
            info = yf.Ticker(ticker)
            news = info.news or []
            headlines = []
            for item in news[:20]:
                title = (
                    item.get("title") or
                    item.get("content", {}).get("title") or
                    ""
                )
                if title:
                    headlines.append(title)
            return headlines
        except Exception:
            return []

    @staticmethod
    def _keyword_score(headlines: list[str]) -> float:
        """Simple keyword counting → score in [-1, 1]."""
        if not headlines:
            return 0.0
        pos = neg = 0
        for headline in headlines:
            words = headline.lower().split()
            pos += sum(1 for w in words if w in POSITIVE_WORDS)
            neg += sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 4)

    def _llm_score(self, ticker: str, headlines: list[str]) -> float:
        """Use CAMEL ChatAgent to score sentiment as a float."""
        import re
        joined = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = (
            f"Analyse the sentiment of these news headlines for {ticker}.\n"
            f"{joined}\n\n"
            "Return ONLY a single float between -1.0 (very negative) "
            "and +1.0 (very positive). No explanation."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            text = response.msgs[0].content.strip()
            match = re.search(r"-?\d+\.?\d*", text)
            if match:
                return max(-1.0, min(1.0, float(match.group())))
        except Exception as e:
            logger.debug(f"LLM sentiment failed for {ticker}: {e}")
        return self._keyword_score(headlines)

    def _build_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.0},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Financial Sentiment Analyst",
                content=(
                    "You are a financial sentiment analyst. "
                    "You output ONLY a single float between -1.0 and 1.0."
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL sentiment agent: {e}")
            return None
