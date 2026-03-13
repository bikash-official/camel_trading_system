"""
run_trading.py
─────────────────────────────────────────────────────────────────
Main entry point for the CAMEL Multi-Agent Trading System.

Usage:
  python run_trading.py
  python run_trading.py --tickers AAPL TSLA NVDA SPY BTC-USD
  python run_trading.py --offline
  python run_trading.py --cash 500000 --period 1y
  python run_trading.py --camel-mode    # enables all CAMEL LLM features
  python run_trading.py --backtest      # run backtesting mode
  python run_trading.py --loop          # continuous self-learning loop
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # load OPENAI_API_KEY etc. from .env

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/trading_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)

from core.workflow import TradingWorkflow

# ── Default tickers to run if none specified ──────────────────────────────────
from agents.data_agent import DEFAULT_TICKERS


def run_pipeline(tickers, args):
    """Run one iteration of the 10-agent pipeline."""
    workflow = TradingWorkflow(
        tickers=tickers,
        period=args.period,
        initial_cash=args.cash,
        use_camel_commentary=args.camel_mode,
        use_camel_sentiment=args.camel_mode,
        use_camel_rationale=args.camel_mode,
        use_supervisor=args.camel_mode,
    )
    return workflow.run()


def run_backtest(tickers, args):
    """Run backtesting mode."""
    from backtesting.backtest_engine import BacktestEngine

    engine = BacktestEngine(
        tickers=tickers,
        period=args.period,
        initial_cash=args.cash,
    )
    return engine.run()


def run_loop(tickers, args):
    """Continuous self-learning loop."""
    logger = logging.getLogger("SelfLearning")
    interval = args.loop_interval

    logger.info(f"\n{'═' * 60}")
    logger.info(f"  SELF-LEARNING LOOP MODE")
    logger.info(f"  Interval : {interval} seconds between runs")
    logger.info(f"  Tickers  : {len(tickers)} assets")
    logger.info(f"  Press Ctrl+C to stop")
    logger.info(f"{'═' * 60}\n")

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n{'─' * 40}")
        logger.info(f"  ITERATION {iteration}")
        logger.info(f"{'─' * 40}")

        try:
            # 1. Run the trading pipeline
            state = run_pipeline(tickers, args)

            # 2. Log performance summary
            portfolio = state.portfolio
            pnl = portfolio.get("pnl", 0)
            logger.info(
                f"  Iteration {iteration} complete | "
                f"Value: ${portfolio.get('total_value', 0):,.2f} | "
                f"P&L: ${pnl:+,.2f}"
            )

            # 3. Retrain RL model periodically (every 5 iterations)
            if iteration % 5 == 0:
                logger.info("  🧠 Periodic RL model retrain …")
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "train_rl.py",
                         "--online",
                         "--tickers"] + tickers[:5] +
                        ["--timesteps", "2000"],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        logger.info("  ✓ RL model retrained successfully")
                    else:
                        logger.warning(f"  ✗ RL retrain failed: {result.stderr[-200:]}")
                except Exception as e:
                    logger.warning(f"  ✗ RL retrain error: {e}")

        except KeyboardInterrupt:
            logger.info(f"\n  Stopped after {iteration} iterations.")
            break
        except Exception as e:
            logger.error(f"  ✗ Iteration {iteration} error: {e}")

        logger.info(f"  Sleeping {interval}s before next iteration …")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            logger.info(f"\n  Stopped after {iteration} iterations.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="CAMEL Multi-Agent AI Trading System (10 Agents)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading.py
  python run_trading.py --tickers AAPL TSLA NVDA
  python run_trading.py --cash 500000 --period 1y
  python run_trading.py --camel-mode          # needs OPENAI_API_KEY
  python run_trading.py --backtest            # historical backtesting
  python run_trading.py --loop                # continuous self-learning
        """,
    )
    parser.add_argument("--tickers",       nargs="+", default=None,
                        help="Ticker symbols to analyse (default: built-in list)")
    parser.add_argument("--period",        default="5y",
                        help="Historical data period (1mo, 3mo, 6mo, 1y, 2y, 5y)")
    parser.add_argument("--cash",          type=float, default=100_000,
                        help="Starting paper portfolio cash (default: 100,000)")
    parser.add_argument("--offline",       action="store_true",
                        help="Skip live download, use cached CSV data only")
    parser.add_argument("--camel-mode",    action="store_true",
                        help="Enable all CAMEL LLM features (needs API key)")
    parser.add_argument("--backtest",      action="store_true",
                        help="Run backtesting mode on historical data")
    parser.add_argument("--loop",          action="store_true",
                        help="Run continuous self-learning loop")
    parser.add_argument("--loop-interval", type=int, default=300,
                        help="Seconds between loop iterations (default: 300)")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print("\n" + "═" * 70)
    print("  🐫  CAMEL MULTI-AGENT AI TRADING SYSTEM  (10 Agents)")
    print(f"  Tickers   : {len(tickers)} assets")
    print(f"  Period    : {args.period}")
    print(f"  Cash      : ${args.cash:,.0f}")
    print(f"  CAMEL LLM : {'✓ Enabled' if args.camel_mode else '✗ Disabled (keyword fallbacks active)'}")
    print(f"  Mode      : {'Backtest' if args.backtest else ('Self-Learning Loop' if args.loop else 'Single Run')}")
    print("═" * 70 + "\n")

    # ── Mode selection ─────────────────────────────────────────────────────
    if args.backtest:
        run_backtest(tickers, args)
    elif args.loop:
        run_loop(tickers, args)
    else:
        state = run_pipeline(tickers, args)

        # ── Exit summary ──────────────────────────────────────────────────
        if state.errors:
            print(f"\n⚠  {len(state.errors)} error(s) occurred during the run.")
        else:
            print("\n✓  All 10 agent steps completed successfully.")

        return state


if __name__ == "__main__":
    main()
