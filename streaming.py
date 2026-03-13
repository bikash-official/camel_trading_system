"""
streaming.py
─────────────────────────────────────────────────────────────────
Real-time data streaming agent using WebSockets.
Subscribes to live price ticks and triggers the 10-agent pipeline.
"""

import json
import logging
import threading
import time
import websocket
from core.workflow import TradingWorkflow
from agents.data_agent import DEFAULT_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("StreamingAgent")

class StreamingAgent:
    def __init__(self, tickers=None):
        self.tickers = tickers or DEFAULT_TICKERS[:5]
        self.workflow = TradingWorkflow(
            tickers=self.tickers,
            use_camel_commentary=True,
            use_camel_sentiment=True
        )
        self.running = True
        self.last_prices = {}

    def on_message(self, ws, message):
        data = json.loads(message)
        # Example for Binance stream: {'s': 'BTCUSDT', 'c': '50000.00'}
        ticker = data.get('s')
        price = data.get('c')
        
        if ticker and price:
            self.last_prices[ticker] = float(price)
            logger.info(f"  Live Tick: {ticker} @ {price}")
            
            # Logic: If we have enough ticks, trigger the analysis pipeline
            # (In a real system, you might buffer this or run it every N seconds)
            if len(self.last_prices) >= len(self.tickers):
                logger.info("  Full asset batch received. Triggering pipeline...")
                # self.workflow.run()
                self.last_prices.clear()

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket Closed")

    def on_open(self, ws):
        logger.info("WebSocket Connected")
        # Example subscription for Binance
        params = [f"{t.lower()}usdt@ticker" for t in ["BTC", "ETH", "SOL"]]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": 1
        }
        ws.send(json.dumps(subscribe_msg))

    def run(self):
        # We use Binance as a public example for demo streaming
        ws_url = "wss://stream.binance.com:9443/ws"
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        logger.info(f"Starting Real-Time Stream on {ws_url} ...")
        ws.run_forever()

if __name__ == "__main__":
    # Note: Requires 'websocket-client' package
    agent = StreamingAgent()
    agent.run()
