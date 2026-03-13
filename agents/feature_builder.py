"""
agents/feature_builder.py
─────────────────────────────────────────────────────────────────
Feature Builder Agent.

Converts the output of AnalysisAgent + SentimentAgent into a
normalised 7-element observation vector per ticker that the PPO
model can consume directly:

  [price_norm, sma_norm, rsi_norm, stoch_k, stoch_d, sentiment, vol_norm]

All values are clipped to a reasonable range and filled to 0.5
when missing (neutral assumption).
"""

from __future__ import annotations
import logging
import numpy as np

from core.state import AgentState

logger = logging.getLogger("FeatureBuilder")


class FeatureBuilderAgent:
    """Builds normalised feature vectors for the RL agent."""

    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[FeatureBuilder] Building feature vectors for {len(state.tickers)} tickers …")
        vectors: dict = {}

        for ticker in state.tickers:
            data = state.analyzed_data.get(ticker, {})
            sent = state.sentiment_scores.get(ticker, 0.0)

            fv = self._build_vector(data, sent)
            vectors[ticker] = fv
            logger.info(f"  {ticker:15s} → {[round(x,3) for x in fv]}")

        state.feature_vectors = vectors
        logger.info(f"[FeatureBuilder] Done.\n")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_vector(data: dict, sentiment: float) -> list[float]:
        price   = data.get("current_price") or 1.0
        sma20   = data.get("sma_20")   or price
        rsi     = data.get("rsi_14")   or 50.0
        stk     = data.get("stoch_k")  or 50.0
        std     = data.get("stoch_d")  or 50.0
        volume  = data.get("volume")   or 1.0
        avg_vol = data.get("avg_volume") or (volume or 1.0)

        # Normalise
        price_norm = float(np.clip(price / sma20 if sma20 else 1.0, 0.5, 2.0))
        sma_norm   = float(np.clip(price / sma20 if sma20 else 1.0, 0.5, 2.0))
        rsi_norm   = float(np.clip(rsi / 100.0,   0.0, 1.0))
        stk_norm   = float(np.clip(stk / 100.0,   0.0, 1.0))
        std_norm   = float(np.clip(std / 100.0,   0.0, 1.0))
        sent_norm  = float(np.clip(sentiment,     -1.0, 1.0))
        vol_norm   = float(np.clip(volume / (avg_vol or 1.0), 0.0, 5.0))

        return [price_norm, sma_norm, rsi_norm, stk_norm, std_norm, sent_norm, vol_norm]
