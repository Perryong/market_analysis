"""
Stock Adapter for TraderXO Analysis
Converts yfinance stock data to work with traderXO strategies
"""

import pandas as pd
from typing import Tuple
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels
from traderXO.crypto_signal_analyzer import CryptoSignalAnalyzer, CryptoSignal


def analyze_stock_with_traderxo(ticker: str, market_data_df: pd.DataFrame, 
                                weekly_df: pd.DataFrame = None) -> CryptoSignal:
    """
    Analyze a stock using traderXO methodology
    
    Args:
        ticker: Stock ticker symbol
        market_data_df: Historical OHLCV data from yfinance/seeking_alpha
        weekly_df: Weekly timeframe data (optional)
        
    Returns:
        CryptoSignal with traderXO analysis
    """
    # Prepare dataframe - ensure it has the right column names
    df = market_data_df.copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Standardize column names (yfinance uses Title case, traderXO uses lowercase)
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    # Rename if needed
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure we have lowercase columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate indicators using traderXO methods
    if weekly_df is not None:
        weekly_df = weekly_df.copy()
        # Ensure weekly_df also has DatetimeIndex
        if not isinstance(weekly_df.index, pd.DatetimeIndex):
            weekly_df.index = pd.to_datetime(weekly_df.index)
        # Standardize weekly columns too
        for old_col, new_col in column_mapping.items():
            if old_col in weekly_df.columns:
                weekly_df[new_col] = weekly_df[old_col]
    
    df = TechnicalIndicators.add_all_indicators(df, weekly_df)
    
    # Calculate market profile
    market_profile = KeyLevels.market_profile(df)
    df['poc'] = market_profile['poc']
    df['vah'] = market_profile['vah']
    df['val'] = market_profile['val']
    
    # Analyze using crypto signal analyzer (works same for stocks)
    analyzer = CryptoSignalAnalyzer()
    signal = analyzer.analyze(ticker, df, weekly_df if weekly_df is not None else df)
    
    return signal


def get_stock_data_for_traderxo(stock_market_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract dataframes from MarketData object for traderXO analysis
    
    Args:
        stock_market_data: MarketData object from stock analysis
        
    Returns:
        Tuple of (main_df, weekly_df)
    """
    # Get the historical dataframe
    df = stock_market_data.historical.copy()
    
    # Ensure index is DatetimeIndex (required for TraderXO)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            # Try to find a date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                df = df.set_index(date_cols[0])
            else:
                raise ValueError("Cannot find date column to set as index")
    
    # Ensure index is datetime type
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # For weekly data, resample the historical data
    # Make sure we're working with a DatetimeIndex
    df_temp = df.copy()
    
    # Resample to weekly
    weekly_df = pd.DataFrame()
    
    # Handle both uppercase and lowercase column names
    if 'Open' in df_temp.columns:
        weekly_df['open'] = df_temp['Open'].resample('W').first()
        weekly_df['high'] = df_temp['High'].resample('W').max()
        weekly_df['low'] = df_temp['Low'].resample('W').min()
        weekly_df['close'] = df_temp['Close'].resample('W').last()
        weekly_df['volume'] = df_temp['Volume'].resample('W').sum()
    else:
        weekly_df['open'] = df_temp['open'].resample('W').first()
        weekly_df['high'] = df_temp['high'].resample('W').max()
        weekly_df['low'] = df_temp['low'].resample('W').min()
        weekly_df['close'] = df_temp['close'].resample('W').last()
        weekly_df['volume'] = df_temp['volume'].resample('W').sum()
    
    weekly_df = weekly_df.dropna()
    
    return df, weekly_df

