# 🐫 CAMEL Multi-Agent AI Trading System

A self-learning, **10-agent** stock trading system built on the **CAMEL AI**
framework with **Yahoo Finance** data and **PPO Reinforcement Learning**.

> ⚠️ For educational purposes only. Not financial advice.

---

## 🏗️ Architecture  (10-Agent Pipeline)

```
                    Yahoo Finance Data (100 Companies)
                                │
                                ▼
                     1. Data Collector Agent  (CAMEL ChatAgent)
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    2. Indicator Agent   3. Pattern Agent    5. Sentiment Agent
       (CAMEL ChatAgent)    (CAMEL ChatAgent)   (CAMEL ChatAgent)
            │                   │                   │
            │                   ▼                   │
            │           4. Trend Agent              │
            │              (CAMEL ChatAgent)         │
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                     6. Strategy Agent  (CAMEL ChatAgent)
                                │
                                ▼
                     7. Feature Builder
                                │
                                ▼
              8. Reinforcement Learning Agent (PPO)
                                │
                                ▼
                     9. Risk Agent (CAMEL ChatAgent)
                                │
                                ▼
              10. Execution Agent + Portfolio Manager + Report Agent
```

### All 10 Agents

| # | Agent | CAMEL Role | Fallback (no API key) |
|---|---|---|---|
| 1 | DataCollectionAgent | Filters tickers by data quality | Keep all valid tickers |
| 2 | TechnicalAnalysisAgent | Deduces BUY/SELL/HOLD from data | Python indicator logic |
| 3 | PatternAgent | Deduces BULLISH/BEARISH/NEUTRAL bias | Python pattern detection |
| 4 | TrendAgent | Deduces trend direction & strength | Python ADX/MA logic |
| 5 | SentimentAnalysisAgent | LLM scores news headlines (-1 to +1) | Keyword scoring |
| 6 | StrategyAgent | Bull (Indicators) vs Bear (Patterns) Debate | Weighted signal combo |
| 7 | FeatureBuilderAgent | N/A (no LLM needed) | N/A |
| 8 | RLTradingAgent | Portfolio-wide decision engine | Rule-based fallback |
| 9 | RiskAgent | Risk assessment summary | Position/drawdown limits |
| 10 | ExecutionAgent + PortfolioManager | N/A | Paper trading + analytics |

---

## 📁 Project Structure

```
camel_trading/
├── agents/
│   ├── data_agent.py           ← Downloads OHLCV (100 tickers, 5y history)
│   ├── analysis_agent.py       ← SMA, RSI, MACD, Bollinger, Stochastic
│   ├── pattern_agent.py        ← Head & Shoulders, Double Top, Triangle, Flags
│   ├── trend_agent.py          ← MA crossover, ADX, trend strength/duration
│   ├── sentiment_agent.py      ← Keyword + optional LLM sentiment
│   ├── strategy_agent.py       ← Advanced Debate (Bull vs Bear vs Judge)
│   ├── feature_builder.py      ← Builds RL feature vector
│   ├── rl_agent.py             ← Multi-Asset PPO model predictions
│   ├── risk_agent.py           ← Portfolio risk + VaR + trade veto
│   ├── execution_agent.py      ← Paper trade execution + risk rules
│   ├── portfolio_manager.py    ← Sharpe, drawdown, win rate, P&L
│   └── report_agent.py         ← Final report generation
├── core/
│   ├── state.py                ← Shared AgentState dataclass
│   └── workflow.py             ← 10-agent pipeline orchestrator
├── envs/
│   └── trading_env.py          ← Multi-Asset Portfolio Gymnasium environment
├── backtesting/
│   └── backtest_engine.py      ← Historical backtesting engine
├── data/                       ← Auto-created: CSV cache for tickers
├── models/                     ← Auto-created: trained PPO .zip files
├── logs/                       ← Auto-created: run logs + reports
├── train_rl.py                 ← Multi-Asset PPO training script
├── run_trading.py              ← Main pipeline entry point (5y default)
├── streaming.py                ← Real-time WebSocket streaming client
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd camel_trading

python -m venv venv
# Windows:
.\venv\Scripts/activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

CAMEL AI can use any LLM (OpenAI, Anthropic, Groq, Gemini, etc.).
The system works **without an API key** using built-in fallbacks.

```bash
cp .env.example .env
# Edit .env and add your key:
# GEMINI_API_KEY=AIzaSyB-...
```

### 3. Train the RL Agent

Train the Multi-Asset PPO model:

```bash
# Quick training (2 minutes, 5 assets)
python train_rl.py --tickers AAPL MSFT NVDA TSLA JPM --timesteps 10000

# Continuous Online training (loads model and continues session)
python train_rl.py --online --timesteps 5000

# Full market training (5 years history)
python train_rl.py --period 5y --timesteps 100000
```

### 4. Run the 10-Agent Trading Pipeline

```bash
# Default (100 tickers, 6 months data, 10-agent pipeline)
python run_trading.py

# Custom tickers
python run_trading.py --tickers AAPL TSLA NVDA SPY

# With CAMEL LLM features (needs API key)
python run_trading.py --camel-mode

# Custom cash & period
python run_trading.py --cash 500000 --period 1y
```

### 5. Run Backtesting

```bash
# Backtest on historical data
python run_trading.py --backtest --tickers AAPL SPY NVDA --period 2y

# Backtest with more tickers
python run_trading.py --backtest --tickers AAPL MSFT GOOGL NVDA TSLA JPM SPY QQQ --period 2y
```

### 6. Continuous Self-Learning Loop

```bash
# Run continuous loop (retrains existing RL model incrementally every 5 iterations)
python run_trading.py --loop

# Custom interval (every 10 minutes)
python run_trading.py --loop --loop-interval 600
```

---

## 🤖 How CAMEL Is Used

CAMEL's `ChatAgent` is integrated at **every stage** of the 10-agent pipeline:

### Example CAMEL agent creation:
```python
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,
    model_type=ModelType.GEMINI_1_5_FLASH,
    model_config_dict={"temperature": 0.0},
)
agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Market Analyst",
        content="You are a senior market analyst..."
    ),
    model=model,
)
```

### Agent-to-Agent Communication:

```python
# Agents communicate through the shared AgentState object
data_agent = DataCollectionAgent()          # Step 1
indicator_agent = TechnicalAnalysisAgent()  # Step 2
pattern_agent = PatternAgent()              # Step 3
trend_agent = TrendAgent()                  # Step 4
sentiment_agent = SentimentAnalysisAgent()  # Step 5
strategy_agent = StrategyAgent()            # Step 6
feature_builder = FeatureBuilderAgent()     # Step 7
rl_agent = RLTradingAgent()                 # Step 8
risk_agent = RiskAgent()                    # Step 9
execution_agent = ExecutionAgent()          # Step 10

# Each agent reads from and writes to state
state = data_agent.run(state)        # → state.raw_data
state = indicator_agent.run(state)   # → state.analyzed_data
state = pattern_agent.run(state)     # → state.pattern_signals
state = trend_agent.run(state)       # → state.trend_signals
state = sentiment_agent.run(state)   # → state.sentiment_scores
state = strategy_agent.run(state)    # → state.strategy_signals
state = feature_builder.run(state)   # → state.feature_vectors
state = rl_agent.run(state)          # → state.decisions
state = risk_agent.run(state)        # → state.risk_assessments (can veto trades)
state = execution_agent.run(state)   # → state.portfolio
```

---

## 🧠 Reinforcement Learning Details

### Environment (envs/trading_env.py)
- **Observation**: 7-element float32 vector per timestep
  `[price_norm, sma_norm, rsi_norm, stoch_k, stoch_d, sentiment, vol_norm]`
- **Actions**: `0=HOLD`, `1=BUY`, `2=SELL`
- **Reward**: % portfolio value change per step − small trading penalty
- **Initial cash**: $100,000

### PPO Hyperparameters
| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | MlpPolicy (3-layer MLP) |
| Learning rate | 3e-4 |
| Batch size | 64 |
| N epochs | 10 |
| Gamma (discount) | 0.99 |
| Entropy coefficient | 0.01 |

### Self-Learning Loop
```
while True:
    1. Collect data          (DataCollectionAgent)
    2. Analyse indicators    (TechnicalAnalysisAgent)
    3. Detect patterns       (PatternAgent)
    4. Evaluate trend        (TrendAgent)
    5. Score sentiment       (SentimentAnalysisAgent)
    6. Combine strategy      (StrategyAgent)
    7. Build features        (FeatureBuilderAgent)
    8. RL agent decides      (RLTradingAgent)
    9. Assess risk           (RiskAgent)
    10. Execute & report     (ExecutionAgent + PortfolioManager)

    → Calculate reward
    → Periodically retrain RL model
    → Sleep(interval)
```

---

## 📊 Technical Indicators

| Indicator | Window | Trading Signal |
|---|---|---|
| SMA-20 / SMA-50 | 20 / 50 periods | Price > SMA = Bullish trend |
| EMA-12 / EMA-26 | 12 / 26 periods | Basis for MACD |
| MACD | 12-26-9 | MACD > Signal = Bullish crossover |
| RSI-14 | 14 periods | < 35 = Oversold (buy zone) / > 65 = Overbought |
| Stochastic %K/%D | 14 / 3 periods | < 25 = Oversold / > 75 = Overbought |
| Bollinger Bands | 20 periods ± 2σ | Price at lower band = potential reversal |
| ATR-14 | 14 periods | Volatility measure for position sizing |
| ADX-14 | 14 periods | > 25 = Trending / < 25 = Ranging |

## 📐 Chart Patterns Detected

| Pattern | Type | Signal |
|---|---|---|
| Head and Shoulders | Reversal | Bearish |
| Double Top | Reversal | Bearish |
| Double Bottom | Reversal | Bullish |
| Ascending Triangle | Continuation | Bullish |
| Descending Triangle | Continuation | Bearish |
| Symmetrical Triangle | Continuation | Neutral |
| Bull Flag | Continuation | Bullish |
| Bear Flag | Continuation | Bearish |
| Golden Cross | MA crossover | Bullish |
| Death Cross | MA crossover | Bearish |

---

## 🛡️ Risk Management

- **Max 5%** of total portfolio per single position (RiskAgent)
- **Max 2%** per individual trade (ExecutionAgent)
- **No short selling** (long-only strategy)
- **Transaction costs** of 0.1% modelled on every trade
- **Price validation**: rejects trades with zero/negative prices
- **Trade veto**: RiskAgent can block trades exceeding risk limits
- **Drawdown monitoring**: Warns when portfolio drawdown exceeds 15%
- **VaR calculation**: 95% Value at Risk from historical returns
- **Offline fallback**: uses cached CSV data if Yahoo Finance is unreachable

---

## 📊 Backtesting Metrics

| Metric | Description |
|---|---|
| Total Return | Overall percentage return |
| Sharpe Ratio | Risk-adjusted return (annualised) |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss ratio |
| Total Trades | Number of buy + sell transactions |

---

## 📄 Sample Output

```
══════════════════════════════════════════════════════════════════════
        🤖  CAMEL MULTI-AGENT TRADING SYSTEM  —  REPORT
        10-Agent Pipeline  •  CAMEL AI + PPO Reinforcement Learning
        2026-03-13  14:30:00
══════════════════════════════════════════════════════════════════════

── SUMMARY ─────────────────────────────────────────────────────────
  Assets Analysed   : 100
  Signal Distribution: BUY=41  SELL=32  HOLD=27
  Trades Executed   : 18

── MARKET OVERVIEW ─────────────────────────────────────────────────
  Overall Mood : Bullish 📈
  Bullish       : 8
  Bearish       : 4
  Neutral       : 3

── PATTERN ANALYSIS ────────────────────────────────────────────────
  NVDA            | BULLISH  | Bull Flag (Bullish), Golden Cross (Bullish)
  AAPL            | NEUTRAL  | None

── TREND ANALYSIS ──────────────────────────────────────────────────
  NVDA            | UPTREND    | ADX=35.2 | STRONG | 42d
  TSLA            | DOWNTREND  | ADX=28.1 | MODERATE | 15d

── STRATEGY SIGNALS ────────────────────────────────────────────────
  NVDA            | STRONG_BUY   | conf=0.82 | score=+0.645
  AAPL            | BUY          | conf=0.55 | score=+0.312

── RISK ASSESSMENT ─────────────────────────────────────────────────
  Portfolio Risk   : LOW
  Drawdown         : 0.00%
  Max Concentration: 2.0%
  VaR (95%)        : $1,234

── PORTFOLIO ANALYTICS ─────────────────────────────────────────────
  Sharpe Ratio    : 1.234
  Max Drawdown    : 2.50%
  Win Rate        : 66.7%
  Profit Factor   : 2.15

── AGENTS USED ─────────────────────────────────────────────────────
   1. DataCollectionAgent      (CAMEL ChatAgent)
   2. TechnicalAnalysisAgent   (CAMEL ChatAgent)
   3. PatternAgent             (CAMEL ChatAgent)
   4. TrendAgent               (CAMEL ChatAgent)
   5. SentimentAnalysisAgent   (CAMEL ChatAgent)
   6. StrategyAgent            (CAMEL ChatAgent)
   7. FeatureBuilderAgent
   8. RLTradingAgent           (PPO + CAMEL ChatAgent)
   9. RiskAgent                (CAMEL ChatAgent)
  10. ExecutionAgent + PortfolioManager + ReportAgent
══════════════════════════════════════════════════════════════════════
```

---

## 🔮 Extending the System

| Feature | How to add |
|---|---|
| New LLM provider | Change `ModelPlatformType` in any agent (Anthropic, Groq, etc.) |
| New indicator | Add method to `TechnicalAnalysisAgent._analyse()` |
| New chart pattern | Add detection method to `PatternAgent` |
| New ticker universe | Edit `DEFAULT_TICKERS` in `agents/data_agent.py` |
| Strategy weights | Adjust `DEFAULT_WEIGHTS` in `agents/strategy_agent.py` |
| Risk limits | Edit constants in `agents/risk_agent.py` |
| Options/Futures | Extend `ExecutionAgent` with new action types |
| Live trading | Replace `ExecutionAgent` with a broker API (Alpaca, Zerodha) |
| FinBERT sentiment | Replace keyword scoring in `SentimentAnalysisAgent` |

---

## 📄 License

Apache 2.0 — same as CAMEL AI framework.
Educational use only. Not financial advice.
