"""
train_rl.py
─────────────────────────────────────────────────────────────────
Train the PPO Reinforcement Learning agent.

Usage examples:
  python train_rl.py
  python train_rl.py --tickers AAPL TSLA NVDA --timesteps 100000
  python train_rl.py --ticker SPY --timesteps 50000
  python train_rl.py --online --timesteps 20000

The trained model is saved to models/ppo_trading_agent.zip and can
be loaded immediately by RLTradingAgent in the main pipeline.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Trainer")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "ppo_trading_agent.zip"

# ── Default training tickers ──────────────────────────────────────────────────
DEFAULT_TRAIN_TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA", "JPM", "BTC-USD"]


def download_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Download historical data with local CSV fallback."""
    cache = Path("data") / f"{ticker.replace('/', '_')}.csv"
    Path("data").mkdir(exist_ok=True)

    try:
        logger.info(f"  Downloading {ticker} …")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache)
            logger.info(f"  ✓ {ticker}: {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"  Download failed for {ticker}: {e}")

    if cache.exists():
        logger.info(f"  Loading {ticker} from cache …")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    return None


def build_combined_env(tickers: list[str], period: str = "2y"):
    """
    Build a gym environment for multi-asset training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.trading_env import TradingEnv

    dfs = {}
    for ticker in tickers:
        df = download_data(ticker, period=period)
        if df is None or len(df) < 60:
            logger.warning(f"  Skipping {ticker}: insufficient data")
            continue
        dfs[ticker] = df

    if not dfs:
        raise ValueError("No valid training data found!")

    logger.info(f"  Building TradingEnv with {len(dfs)} assets …")
    return DummyVecEnv([lambda dfs=dfs: TradingEnv(dfs)])


def train(
    tickers: list[str],
    timesteps: int = 50_000,
    period: str = "2y",
    learning_rate: float = 3e-4,
    online: bool = False,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    logger.info(f"\n{'═'*60}")
    logger.info(f"  TRAINING PPO RL AGENT")
    logger.info(f"  Tickers    : {tickers}")
    logger.info(f"  Timesteps  : {timesteps:,}")
    logger.info(f"  Period     : {period}")
    logger.info(f"  Online     : {online}")
    logger.info(f"{'═'*60}\n")

    vec_env = build_combined_env(tickers, period=period)

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    if online and MODEL_PATH.exists():
        logger.info(f"  Loading existing model from {MODEL_PATH} for online learning...")
        model = PPO.load(str(MODEL_PATH), env=vec_env)
        # update learning rate if we want to change it
        model.learning_rate = learning_rate
    else:
        if online:
            logger.info("  Online mode requested but no existing model found. Training from scratch.")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # small entropy bonus encourages exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )

    # ── Checkpoint callback ───────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(timesteps // 10, 1000),
        save_path=str(MODEL_DIR),
        name_prefix="ppo_checkpoint",
    )

    logger.info("  Starting training …\n")
    model.learn(total_timesteps=timesteps, callback=checkpoint_cb, reset_num_timesteps=not online)

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    logger.info(f"\n  ✓ Model saved to {MODEL_PATH}")

    # ── Quick evaluation ──────────────────────────────────────────────────────
    _evaluate(model, vec_env)

    return model


def _evaluate(model, vec_env, n_episodes: int = 5):
    """Run n_episodes and print average reward."""
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_episodes)
    logger.info(f"\n  Evaluation over {n_episodes} episodes:")
    logger.info(f"  Mean reward : {mean_reward:.4f} ± {std_reward:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO RL trading agent")
    parser.add_argument("--tickers",    nargs="+", default=DEFAULT_TRAIN_TICKERS,
                        help="List of Yahoo Finance tickers to train on")
    parser.add_argument("--ticker",     type=str, default=None,
                        help="Single ticker shorthand (e.g. --ticker SPY)")
    parser.add_argument("--timesteps",  type=int, default=50_000,
                        help="Total training timesteps (default: 50,000)")
    parser.add_argument("--period",     type=str, default="2y",
                        help="Historical data window (e.g. 1y, 2y, 5y)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--online",     action="store_true",
                        help="Continue training from an existing model (default: train from scratch)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else args.tickers

    try:
        train(
            tickers=tickers,
            timesteps=args.timesteps,
            period=args.period,
            learning_rate=args.lr,
            online=args.online,
        )
    except ImportError as e:
        logger.error(f"\n  Missing dependency: {e}")
        logger.error("  Run:  pip install -r requirements.txt")
        sys.exit(1)
