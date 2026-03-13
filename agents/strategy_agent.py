"""
agents/strategy_agent.py
─────────────────────────────────────────────────────────────────
Strategy Agent — the "brain" that combines all analysis.

Receives signals from:
  • TechnicalAnalysisAgent  (indicators)
  • PatternAgent            (chart patterns)
  • SentimentAgent          (news sentiment)
  • TrendAgent              (trend direction / strength)

Outputs a weighted composite signal per ticker:
  STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL  +  confidence

A CAMEL ChatAgent can optionally provide strategy reasoning.
"""

from __future__ import annotations
import logging
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState

logger = logging.getLogger("StrategyAgent")

# ── Default signal weights ────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "indicators": 0.30,
    "patterns":   0.20,
    "sentiment":  0.20,
    "trend":      0.30,
}


class StrategyAgent:
    """
    Combines signals from multiple analysis agents into a unified
    strategy signal. Results stored in state.strategy_signals.
    """

    def __init__(self, weights: dict | None = None, use_camel_reasoning: bool = False):
        self.weights = weights or DEFAULT_WEIGHTS
        self.use_camel_reasoning = use_camel_reasoning
        
        if use_camel_reasoning:
            self._bull_agent = self._build_agent(
                "Bullish Analyst", 
                "You are an aggressive bull analyst analyzing Indicator and Sentiment data. Find reasons why this stock is a BUY. Focus on positive signals. Give a 2-sentence bullish argument and end with exactly 'Argument: BUY'."
            )
            self._bear_agent = self._build_agent(
                "Bearish Analyst", 
                "You are an aggressive bear analyst analyzing Pattern and Trend data. Find reasons why this stock is a SELL. Focus on negative signals. Give a 2-sentence bearish argument and end with exactly 'Argument: SELL'."
            )
            self._judge_agent = self._build_agent(
                 "Strategy Judge", 
                 "You are the head trading judge. Read the Bull argument (Indicators+Sentiment) and Bear argument (Patterns+Trend). Weigh them impartially and decide the final strategy in 2 sentences. Your response MUST start with Final Signal: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]"
            )
        else:
            self._bull_agent = None
            self._bear_agent = None
            self._judge_agent = None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[StrategyAgent] Combining signals for {len(state.tickers)} tickers …")
        signals: dict = {}

        for ticker in state.tickers:
            try:
                base_result = self._combine_signals(ticker, state)
                if self._bull_agent and self._bear_agent and self._judge_agent:
                    debate_result = self._run_debate(ticker, base_result, state)
                    signals[ticker] = debate_result
                else:
                    signals[ticker] = base_result
            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")
                state.errors.append(f"StrategyAgent/{ticker}: {e}")

        state.strategy_signals = signals
        logger.info(f"[StrategyAgent] Done. {len(signals)} strategies generated.\n")
        return state

    # ── Signal combination ────────────────────────────────────────────────────
    def _combine_signals(self, ticker: str, state: AgentState) -> dict:
        w = self.weights

        # ── Indicator score  (from AnalysisAgent signal) ─────────────────────
        analysis = state.analyzed_data.get(ticker, {})
        indicator_signal = analysis.get("signal", "HOLD")
        indicator_score = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(indicator_signal, 0.0)

        rsi = analysis.get("rsi_14", 50)
        # Boost score for extreme RSI
        if rsi and rsi < 30:
            indicator_score += 0.5
        elif rsi and rsi > 70:
            indicator_score -= 0.5

        # ── Pattern score ────────────────────────────────────────────────────
        pattern_data = state.pattern_signals.get(ticker, {})
        pattern_bias = pattern_data.get("overall_pattern_bias", "NEUTRAL")
        pattern_score = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(pattern_bias, 0.0)
        # Boost for multiple patterns
        pattern_count = pattern_data.get("pattern_count", 0)
        if pattern_count > 2:
            pattern_score *= 1.3

        # ── Sentiment score ──────────────────────────────────────────────────
        sentiment = state.sentiment_scores.get(ticker, 0.0)
        sentiment_score = max(-1.0, min(1.0, sentiment * 2))  # amplify

        # ── Trend score ──────────────────────────────────────────────────────
        trend_data = state.trend_signals.get(ticker, {})
        trend = trend_data.get("trend", "SIDEWAYS")
        trend_base = {"UPTREND": 1.0, "DOWNTREND": -1.0, "SIDEWAYS": 0.0}.get(trend, 0.0)
        # Weight by strength
        strength = trend_data.get("trend_strength", "WEAK")
        strength_mult = {"STRONG": 1.0, "MODERATE": 0.7, "WEAK": 0.3}.get(strength, 0.5)
        trend_score = trend_base * strength_mult

        # ── Weighted composite ───────────────────────────────────────────────
        composite = (
            w["indicators"] * indicator_score +
            w["patterns"]   * pattern_score +
            w["sentiment"]  * sentiment_score +
            w["trend"]      * trend_score
        )

        # Map composite to signal
        if composite >= 0.5:
            signal = "STRONG_BUY"
        elif composite >= 0.2:
            signal = "BUY"
        elif composite <= -0.5:
            signal = "STRONG_SELL"
        elif composite <= -0.2:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = min(1.0, abs(composite))

        result = {
            "ticker": ticker,
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "composite_score": round(float(composite), 4),
            "components": {
                "indicator_score": round(float(indicator_score), 3),
                "pattern_score": round(float(pattern_score), 3),
                "sentiment_score": round(float(sentiment_score), 3),
                "trend_score": round(float(trend_score), 3),
            },
        }
        logger.info(
            f"  ✓ {ticker:15s} | {signal:12s} | conf={confidence:.2f} | "
            f"score={composite:+.3f}"
        )
        return result

    # ── CAMEL reasoning ───────────────────────────────────────────────────────
    def _build_agent(self, role_name: str, instruction: str) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=ModelType.GEMINI_1_5_FLASH,
                model_config_dict={"temperature": 0.2},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name=role_name,
                content=instruction,
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL agent {role_name}: {e}")
            return None

    def _run_debate(self, ticker: str, base_result: dict, state: AgentState) -> dict:
        components = base_result["components"]
        
        bull_data_str = (
            f"Ticker: {ticker}\n"
            f"Indicator Score: {components['indicator_score']}\n"
            f"Sentiment Score: {components['sentiment_score']}\n"
        )
        
        bear_data_str = (
            f"Ticker: {ticker}\n"
            f"Pattern Score: {components['pattern_score']}\n"
            f"Patterns: {state.pattern_signals.get(ticker, {}).get('patterns', [])}\n"
            f"Trend Score: {components['trend_score']}\n"
            f"Trend: {state.trend_signals.get(ticker, {}).get('trend', 'N/A')}\n"
        )
        
        full_data_str = (
             f"Ticker: {ticker}\n"
             f"Composite Score: {base_result['composite_score']}\n"
             f"Indicator: {components['indicator_score']} | Sentiment: {components['sentiment_score']}\n"
             f"Pattern: {components['pattern_score']} | Trend: {components['trend_score']}\n"
        )
        
        try:
            # Get Bull argument
            bull_msg = BaseMessage.make_user_message(role_name="User", content=bull_data_str)
            bull_resp = self._bull_agent.step(bull_msg).msgs[0].content.strip()
            
            # Get Bear argument
            bear_msg = BaseMessage.make_user_message(role_name="User", content=bear_data_str)
            bear_resp = self._bear_agent.step(bear_msg).msgs[0].content.strip()
            
            # Get Judge argument
            judge_prompt = (
                f"{full_data_str}\n\n"
                f"--- BULL ARGUMENT ---\n{bull_resp}\n\n"
                f"--- BEAR ARGUMENT ---\n{bear_resp}\n\n"
                "Decide the final strategy."
            )
            judge_msg = BaseMessage.make_user_message(role_name="User", content=judge_prompt)
            judge_resp = self._judge_agent.step(judge_msg).msgs[0].content.strip()
            
            # Parse the judge's signal
            signal = base_result["signal"] # fallback
            first_line = judge_resp.split("\n")[0]
            if "Final Signal:" in first_line:
                signal_part = first_line.split("Final Signal:")[1].strip()
                # find the first word
                signal_word = signal_part.split()[0].upper()
                if signal_word in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
                    signal = signal_word
            
            base_result["signal"] = signal
            base_result["reasoning"] = f"BULL:\n{bull_resp}\n\nBEAR:\n{bear_resp}\n\nJUDGE:\n{judge_resp}"
            return base_result
            
        except Exception as e:
            logger.warning(f"Strategy debate failed for {ticker}: {e}")
            base_result["reasoning"] = f"Debate unavailable: {e}"
            return base_result

    def _get_reasoning(self, ticker: str, result: dict, state: AgentState) -> str:
        components = result["components"]
        prompt = (
            f"Ticker: {ticker}\n"
            f"Final Signal: {result['signal']} (confidence {result['confidence']:.2f})\n"
            f"Composite Score: {result['composite_score']}\n"
            f"Indicator Score: {components['indicator_score']}\n"
            f"Pattern Score: {components['pattern_score']}\n"
            f"Sentiment Score: {components['sentiment_score']}\n"
            f"Trend Score: {components['trend_score']}\n"
            f"Trend: {state.trend_signals.get(ticker, {}).get('trend', 'N/A')}\n"
            f"Patterns: {state.pattern_signals.get(ticker, {}).get('patterns', [])}\n"
            "Explain this strategy in 2-3 sentences."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            # Create an ephemeral reasoning agent since _camel_agent isn't bound to 'self' dynamically
            reasoning_agent = self._build_agent(
                "Strategy Explainer",
                "You are a helpful AI trading core. Explain the final trading signal."
            )
            if reasoning_agent:
                response = reasoning_agent.step(user_msg)
                return response.msgs[0].content.strip()
            return "Strategy reasoning unavailable: Agent creation failed."
        except Exception as e:
            return f"Strategy reasoning unavailable: {e}"
