"""
Technical Indicators Module
Calculate EMAs, VWAP, ATR, and other technical indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """Calculate technical indicators for trading analysis"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def vwap(df: pd.DataFrame, session_start: str = '00:00') -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            df: DataFrame with OHLCV data
            session_start: Time to reset VWAP calculation
            
        Returns:
            VWAP series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Reset VWAP at session start (daily by default)
        df_copy = df.copy()
        
        # Ensure index is DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)
        
        # Extract date from datetime index
        df_copy['date'] = pd.to_datetime(df_copy.index.date)
        
        vwap_list = []
        for date, group in df_copy.groupby('date'):
            tp = (group['high'] + group['low'] + group['close']) / 3
            vol = group['volume']
            
            cumulative_tp_vol = (tp * vol).cumsum()
            cumulative_vol = vol.cumsum()
            
            vwap = cumulative_tp_vol / cumulative_vol
            vwap_list.append(vwap)
        
        return pd.concat(vwap_list)
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def calculate_delta(df: pd.DataFrame, trades_df: pd.DataFrame = None) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate delta (buy volume - sell volume)
        
        Args:
            df: OHLCV DataFrame
            trades_df: Optional trades DataFrame with delta column
            
        Returns:
            Tuple of (delta series, cumulative delta series)
        """
        if trades_df is not None and 'delta' in trades_df.columns:
            # Aggregate trades to candle timeframe
            trades_df['candle_time'] = trades_df['timestamp'].dt.floor(
                pd.infer_freq(df.index)
            )
            delta = trades_df.groupby('candle_time')['delta'].sum()
            delta = delta.reindex(df.index, fill_value=0)
        else:
            # Estimate delta from price action and volume
            # If close > open, assume buying pressure
            delta = df.apply(
                lambda x: x['volume'] if x['close'] > x['open'] else -x['volume'],
                axis=1
            )
        
        cumulative_delta = delta.cumsum()
        return delta, cumulative_delta
    
    @staticmethod
    def relative_strength(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
        """
        Calculate relative strength (ratio of two assets)
        
        Args:
            df1: First asset OHLCV data
            df2: Second asset OHLCV data
            
        Returns:
            Relative strength ratio
        """
        # Align the data
        combined = pd.concat([df1['close'], df2['close']], axis=1, keys=['asset1', 'asset2'])
        combined = combined.fillna(method='ffill')
        
        rs = combined['asset1'] / combined['asset2']
        return rs
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            series: Price series
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal_len: Signal line period (default 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def volume_threshold(df: pd.DataFrame, multiplier: float = 3.0) -> float:
        """
        Calculate volume threshold (average * multiplier)
        
        Args:
            df: OHLCV DataFrame
            multiplier: Multiplier for average volume
            
        Returns:
            Volume threshold value
        """
        avg_volume = df['volume'].mean()
        return avg_volume * multiplier
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add all indicators to DataFrame
        
        Args:
            df: Main timeframe OHLCV data
            weekly_df: Weekly timeframe data for 12/21 EMAs
            
        Returns:
            DataFrame with all indicators
        """
        df = df.copy()
        
        # Intraday EMAs
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        
        # Weekly EMAs (if provided)
        if weekly_df is not None:
            weekly_df = weekly_df.copy()
            weekly_df['ema_12'] = TechnicalIndicators.ema(weekly_df['close'], 12)
            weekly_df['ema_21'] = TechnicalIndicators.ema(weekly_df['close'], 21)
            
            # Resample to main timeframe
            df['ema_12_weekly'] = weekly_df['ema_12'].reindex(df.index, method='ffill')
            df['ema_21_weekly'] = weekly_df['ema_21'].reindex(df.index, method='ffill')
        
        # VWAP
        df['vwap'] = TechnicalIndicators.vwap(df)
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df)
        
        # Delta
        df['delta'], df['cumulative_delta'] = TechnicalIndicators.calculate_delta(df)
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.macd(df['close'])
        
        # Volume metrics
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_threshold'] = TechnicalIndicators.volume_threshold(df)
        
        return df

