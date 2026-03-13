"""
envs/trading_env.py
─────────────────────────────────────────────────────────────────
Custom Gymnasium environment for training the PPO RL agent.
Supports multiple assets in a single environment.

Observation space : 1 (cash) + N_ASSETS * 8 features
    cash + per asset [shares, price_norm, sma20_norm, rsi_norm, 
                      stoch_k_norm, stoch_d_norm, sentiment, volume_norm]

Action space      : MultiDiscrete([3] * N_ASSETS)  →  0=HOLD  1=BUY  2=SELL per asset

Reward            : portfolio_value_change + position_penalty
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        trade_pct: float = 0.10,       # fraction of cash per trade
        transaction_cost: float = 0.001,  # 0.1% per trade
    ):
        super().__init__()

        self.tickers = list(dfs.keys())
        self.n_assets = len(self.tickers)
        
        if self.n_assets == 0:
            raise ValueError("No valid tickers provided to TradingEnv")

        # Find minimum length to align
        self.max_steps = min(len(df) for df in dfs.values())
        self.dfs = {k: v.iloc[:self.max_steps].copy().reset_index(drop=True) for k, v in dfs.items()}

        self.initial_cash = initial_cash
        self.trade_pct = trade_pct
        self.tc = transaction_cost

        # ── Precompute indicators for all assets ──────────────────────────
        self.asset_data = {}
        for ticker, df in self.dfs.items():
            self.asset_data[ticker] = self._precompute(df)

        # ── Gym spaces ────────────────────────────────────────────────────
        # cash + (shares + 7 features) * n_assets
        obs_dim = 1 + self.n_assets * 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)

        # ── State variables ───────────────────────────────────────────────
        self.current_step = 0
        self.cash = initial_cash
        self.shares = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = initial_cash
        self.prev_portfolio_value = initial_cash

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 50  # skip warm-up period
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        return self._obs(), {}

    def step(self, actions: np.ndarray):
        prices = self._prices()
        prev_value = self._portfolio_value(prices)

        reward_penalty = 0.0

        # ── Execute actions ───────────────────────────────────────────────
        for i, action in enumerate(actions):
            price = prices[i]
            if action == 1 and self.cash > 0:          # BUY
                spend = self.cash * self.trade_pct
                shares_bought = spend / (price * (1 + self.tc))
                self.shares[i] += shares_bought
                self.cash -= spend
                reward_penalty -= 0.0005

            elif action == 2 and self.shares[i] > 0:   # SELL
                proceeds = self.shares[i] * price * (1 - self.tc)
                self.cash += proceeds
                self.shares[i] = 0.0
                reward_penalty -= 0.0005

        # ── Advance ───────────────────────────────────────────────────────
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        new_prices = self._prices()
        new_value = self._portfolio_value(new_prices)
        self.portfolio_value = new_value

        # ── Reward ───────────────────────────────────────────────────────
        reward = (new_value - prev_value) / prev_value   # % change
        reward += reward_penalty

        return self._obs(), float(reward), done, False, {"portfolio_value": new_value}

    def render(self, mode="human"):
        print(f"Step {self.current_step:4d} | Cash {self.cash:.0f} | Value {self.portfolio_value:.0f}")

    # ── Internals ─────────────────────────────────────────────────────────────
    def _precompute(self, df: pd.DataFrame):
        close = pd.to_numeric(df["Close"], errors="coerce")
        high  = pd.to_numeric(df.get("High",  close), errors="coerce")
        low   = pd.to_numeric(df.get("Low",   close), errors="coerce")
        vol   = pd.to_numeric(df.get("Volume", pd.Series(np.zeros(len(close)))), errors="coerce")

        sma20 = close.rolling(20).mean()
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        ll   = low.rolling(14).min()
        hh   = high.rolling(14).max()
        stk  = 100 * (close - ll) / (hh - ll + 1e-9)
        std  = stk.rolling(3).mean()

        data = {
            "close": close.values.astype(np.float32),
            "sma20": (sma20 / close).fillna(1.0).values.astype(np.float32),
            "rsi":   (rsi / 100.0).fillna(0.5).values.astype(np.float32),
            "stk":   (stk / 100.0).fillna(0.5).values.astype(np.float32),
            "std":   (std / 100.0).fillna(0.5).values.astype(np.float32),
        }
        vol_mean = vol.rolling(20).mean().replace(0, 1)
        data["vol"] = (vol / vol_mean).fillna(1.0).clip(0, 5).values.astype(np.float32)
        data["sent"] = np.zeros(len(close), dtype=np.float32)
        return data

    def _obs(self) -> np.ndarray:
        i = self.current_step
        obs = [self.cash / self.initial_cash] # Normalized cash
        
        for idx, ticker in enumerate(self.tickers):
            d = self.asset_data[ticker]
            price = d["close"][i]
            price_norm = price / d["close"][max(0, i-50):i+1].mean() if i > 0 else 1.0
            
            obs.extend([
                self.shares[idx] * price / self.initial_cash, # Normalized position value
                price_norm,
                d["sma20"][i],
                d["rsi"][i],
                d["stk"][i],
                d["std"][i],
                d["sent"][i],
                d["vol"][i]
            ])
            
        return np.array(obs, dtype=np.float32)

    def _prices(self) -> np.ndarray:
        i = self.current_step
        return np.array([self.asset_data[t]["close"][i] for t in self.tickers], dtype=np.float32)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return self.cash + float(np.sum(self.shares * prices))
