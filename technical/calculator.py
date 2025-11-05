"""Technical indicator calculator wrapper"""

import pandas as pd
from indicators.index import prepare_indicators
from indicators.candle_sticks.index import prepare_candle_sticks


class TechnicalCalculator:
    """Facade for calculating technical indicators"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and candlestick patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        # Reset index to ensure Date is a column
        df = df.reset_index()
        
        # Calculate technical indicators
        df = prepare_indicators(df)
        
        # Detect candlestick patterns
        df = prepare_candle_sticks(df)
        
        return df

