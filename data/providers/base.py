"""Base protocol for data providers"""

from typing import Protocol, List
import pandas as pd
from core.models import MarketData
from core.enums import TimeFrame


class DataProvider(Protocol):
    """Protocol defining interface for market data providers"""
    
    def fetch_data(self, ticker: str, period: str = "2y", 
                   timeframe: TimeFrame = TimeFrame.DAILY) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1y", "2y", "5y")
            timeframe: Data granularity (daily, weekly, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        ...
    
    def get_market_data(self, ticker: str, period: str = "2y",
                       timeframe: TimeFrame = TimeFrame.DAILY) -> MarketData:
        """
        Fetch and prepare complete market data for analysis
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            timeframe: Data granularity
            
        Returns:
            MarketData object ready for analysis
        """
        ...
    
    def get_ticker_name(self, ticker: str) -> str:
        """
        Get human-readable name for ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company/security name
        """
        ...

