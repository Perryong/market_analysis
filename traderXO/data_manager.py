"""
Data Management Module
Handles fetching OHLCV data from exchanges using CCXT
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time


class DataManager:
    """Fetch and manage trading data from multiple exchanges"""
    
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize data manager with exchange
        
        Args:
            exchange_name: Exchange to use (default: binance)
        """
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange(exchange_name)
    
    def _init_exchange(self, exchange_name: str):
        """Initialize CCXT exchange"""
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        return exchange
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                     lookback_days: int = 30) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            lookback_days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Must be one of: {self.TIMEFRAMES}")
        
        # Calculate since timestamp
        since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        print(f"[*] Fetching {symbol} {timeframe} data from {self.exchange_name}...")
        
        # Fetch data in chunks if needed (exchange limits)
        all_data = []
        limit = 1000  # Most exchanges limit to 1000 candles per request
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Check if we have all data
                if len(ohlcv) < limit:
                    break
                
                # Update since for next chunk
                since = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)  # Rate limiting
                
            except Exception as e:
                print(f"[ERROR] Failed to fetch data: {e}")
                break
        
        if not all_data:
            raise ValueError(f"No data retrieved for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"[OK] Retrieved {len(df)} candles")
        return df
    
    def fetch_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades for delta calculation
        
        Args:
            symbol: Trading pair
            limit: Number of recent trades
            
        Returns:
            DataFrame with trade data
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate delta (buy volume - sell volume)
            df['delta'] = df.apply(
                lambda x: x['amount'] if x['side'] == 'buy' else -x['amount'],
                axis=1
            )
            
            return df
            
        except Exception as e:
            print(f"[WARNING] Could not fetch trades: {e}")
            return pd.DataFrame()
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch current order book
        
        Args:
            symbol: Trading pair
            limit: Depth of order book
            
        Returns:
            Dictionary with bids and asks
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return orderbook
        except Exception as e:
            print(f"[WARNING] Could not fetch order book: {e}")
            return {'bids': [], 'asks': []}
    
    def fetch_multiple_timeframes(self, symbol: str, 
                                  timeframes: List[str],
                                  lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            lookback_days: Lookback period
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv(symbol, tf, lookback_days)
                data[tf] = df
                time.sleep(1)  # Rate limiting between requests
            except Exception as e:
                print(f"[ERROR] Failed to fetch {tf}: {e}")
        
        return data

