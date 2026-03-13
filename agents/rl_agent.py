"""
agents/rl_agent.py
─────────────────────────────────────────────────────────────────
Reinforcement Learning Agent (PPO via stable-baselines3).

This agent:
  1. Loads the trained PPO model from models/ppo_trading_agent.zip
  2. For each ticker builds its 7-element observation vector
  3. Queries the model → returns action (0=HOLD, 1=BUY, 2=SELL)
     and a confidence score (max softmax probability)
  4. Writes results to state.decisions and state.confidence

A CAMEL ChatAgent can be optionally enabled to provide a
human-readable rationale for each trading decision.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from core.state import AgentState

logger = logging.getLogger("RLAgent")

MODEL_PATH = Path("models/ppo_trading_agent.zip")
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


class RLTradingAgent:
    """
    Wraps a trained Stable-Baselines3 PPO model.
    Falls back to rule-based signals when no trained model exists.
    """

    def __init__(self, model_path: Path = MODEL_PATH, use_camel_rationale: bool = False):
        self.model_path = model_path
        self.use_camel_rationale = use_camel_rationale
        self._ppo = self._load_ppo()
        self._camel_agent = self._build_camel_agent() if use_camel_rationale else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"[RLAgent] Making decisions for {len(state.tickers)} tickers …")
        decisions: dict = {}
        confidence: dict = {}

        if self._ppo is not None:
            try:
                # Determine how many assets the model expects
                obs_dim = self._ppo.observation_space.shape[0]
                n_assets = (obs_dim - 1) // 8
                
                if len(state.tickers) == n_assets:
                    obs = [1.0] # Cash (normalized to 1.0)
                    for ticker in state.tickers:
                        fv = state.feature_vectors.get(ticker, [1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 1.0])
                        obs.extend([0.0] + fv) # 0.0 for shares normalized
                    
                    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
                    actions, _ = self._ppo.predict(obs_arr, deterministic=True)
                    
                    if len(actions.shape) == 2:
                        actions = actions[0] # remove batch dim
                        
                    for i, ticker in enumerate(state.tickers):
                        act_int = int(actions[i])
                        decisions[ticker] = ACTIONS[act_int]
                        confidence[ticker] = 0.70 # default multi-discrete confidence
                        
                        rationale = ""
                        if self._camel_agent:
                            rationale = self._get_rationale(ticker, state.feature_vectors.get(ticker, []), ACTIONS[act_int])
                        logger.info(f"  {ticker:15s} → {ACTIONS[act_int]:4s}  conf=0.70" + (f"  | {rationale}" if rationale else ""))

                    state.decisions = decisions
                    state.confidence = confidence
                    logger.info(f"[RLAgent] Multi-asset PPO Done.\n")
                    return state
                else:
                    logger.warning(f"  Mismatch: PPO expects {n_assets} tickers, but {len(state.tickers)} provided. Falling back to rule-based.")
            except Exception as e:
                logger.warning(f"  Multi-asset predict failed: {e}. Falling back to rule-based.")

        # ── Rule-based fallback ───────────────────────────────────────────────
        for ticker in state.tickers:
            fv = state.feature_vectors.get(ticker)
            if fv is None:
                continue

            action, conf = self._rule_based(fv)
            decisions[ticker] = action
            confidence[ticker] = conf

            rationale = ""
            if self._camel_agent:
                rationale = self._get_rationale(ticker, fv, action)

            logger.info(f"  {ticker:15s} → {action:4s}  conf={conf:.2f}" + (f"  | {rationale}" if rationale else ""))

        state.decisions = decisions
        state.confidence = confidence
        logger.info(f"[RLAgent] Done.\n")
        return state

    @staticmethod
    def _rule_based(fv: list[float]) -> tuple[str, float]:
        """
        Simple fallback rules using the 7-element feature vector:
        [price_norm, sma_norm, rsi_norm, stk, std, sentiment, vol_norm]
        """
        _, sma_norm, rsi_norm, stk, std, sentiment, _ = fv
        rsi = rsi_norm * 100

        score = 0
        score += 1  if sma_norm > 1.0     else -1   # price above SMA
        score += 2  if rsi < 35            else 0
        score -= 2  if rsi > 65            else 0
        score += 1  if stk * 100 < 25      else 0    # stochastic oversold
        score -= 1  if stk * 100 > 75      else 0    # stochastic overbought
        score += 1  if sentiment > 0.2     else 0
        score -= 1  if sentiment < -0.2    else 0

        if score >= 2:
            return "BUY",  min(0.5 + score * 0.05, 0.95)
        elif score <= -2:
            return "SELL", min(0.5 + abs(score) * 0.05, 0.95)
        else:
            return "HOLD", 0.55

    def _load_ppo(self):
        if not self.model_path.exists():
            logger.warning(
                f"[RLAgent] No model found at {self.model_path}. "
                "Using rule-based fallback. Run train_rl.py to train."
            )
            return None
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(self.model_path))
            logger.info(f"[RLAgent] Loaded PPO model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"[RLAgent] Failed to load PPO: {e}")
            return None

    # ── CAMEL rationale ───────────────────────────────────────────────────────
    def _build_camel_agent(self) -> ChatAgent | None:
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict={"temperature": 0.2},
            )
            sys_msg = BaseMessage.make_assistant_message(
                role_name="Trading Strategy Explainer",
                content=(
                    "You explain trading decisions in one concise sentence "
                    "based on the given indicator values. Be specific."
                ),
            )
            return ChatAgent(system_message=sys_msg, model=model)
        except Exception as e:
            logger.warning(f"Could not create CAMEL rationale agent: {e}")
            return None

    def _get_rationale(self, ticker: str, fv: list[float], action: str) -> str:
        labels = ["price_norm", "sma_norm", "rsi_norm", "stoch_k", "stoch_d", "sentiment", "volume_norm"]
        vals   = dict(zip(labels, fv))
        prompt = (
            f"Ticker: {ticker}\n"
            f"Action: {action}\n"
            f"Indicators: {vals}\n"
            "Explain this trading decision in one sentence."
        )
        try:
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = self._camel_agent.step(user_msg)
            return response.msgs[0].content.strip()
        except Exception:
            return ""
