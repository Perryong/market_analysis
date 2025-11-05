"""Seeking Alpha data provider implementation"""

import pandas as pd
from core.models import MarketData
from core.enums import TimeFrame
from technical.calculator import TechnicalCalculator
from utils.seeking_alpha import download as sa_download


class SeekingAlphaProvider:
    """Data provider using Seeking Alpha API"""
    
    def __init__(self):
        self.calculator = TechnicalCalculator()
        
    def fetch_data(self, ticker: str, period: str = "2y", 
                   timeframe: TimeFrame = TimeFrame.DAILY) -> pd.DataFrame:
        """
        Fetch OHLCV data from Seeking Alpha
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            timeframe: Data granularity
            
        Returns:
            DataFrame with OHLCV data
        """
        interval = timeframe.value
        
        df = sa_download(
            ticker,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True
        )
        
        return df
    
    def get_market_data(self, ticker: str, period: str = "2y",
                       timeframe: TimeFrame = TimeFrame.DAILY) -> MarketData:
        """
        Fetch and prepare market data with all indicators
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            timeframe: Data granularity
            
        Returns:
            Complete MarketData object
        """
        # Fetch raw data
        df = self.fetch_data(ticker, period, timeframe)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Calculate all technical indicators
        df = self.calculator.calculate_all(df)
        
        # Store ticker for later reference
        df.attrs['TICKER'] = ticker
        
        # Extract current and previous rows
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        return MarketData(
            ticker=ticker,
            current=current,
            previous=previous,
            historical=df,
            timeframe=timeframe,
            short_name=ticker  # Seeking Alpha doesn't provide names easily
        )
    
    def get_ticker_name(self, ticker: str) -> str:
        """Get ticker name (not available in Seeking Alpha free tier)"""
        return ticker

