"""
Key Levels Module
Identify and calculate important price levels for trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class KeyLevels:
    """Calculate key price levels for market structure analysis"""
    
    @staticmethod
    def monthly_opens(df: pd.DataFrame) -> pd.Series:
        """
        Calculate monthly open prices
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Series with monthly opens
        """
        df_copy = df.copy()
        df_copy['month'] = df_copy.index.to_period('M')
        
        monthly_opens = df_copy.groupby('month')['open'].first()
        monthly_opens.index = monthly_opens.index.to_timestamp()
        
        # Forward fill to get monthly open for each timestamp
        result = pd.Series(index=df.index, dtype=float)
        for month_start, open_price in monthly_opens.items():
            mask = (df.index >= month_start) & (df.index < month_start + pd.DateOffset(months=1))
            result[mask] = open_price
        
        return result
    
    @staticmethod
    def quarterly_opens(df: pd.DataFrame) -> pd.Series:
        """
        Calculate quarterly open prices
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Series with quarterly opens
        """
        df_copy = df.copy()
        df_copy['quarter'] = df_copy.index.to_period('Q')
        
        quarterly_opens = df_copy.groupby('quarter')['open'].first()
        quarterly_opens.index = quarterly_opens.index.to_timestamp()
        
        # Forward fill
        result = pd.Series(index=df.index, dtype=float)
        for quarter_start, open_price in quarterly_opens.items():
            mask = (df.index >= quarter_start) & (df.index < quarter_start + pd.DateOffset(months=3))
            result[mask] = open_price
        
        return result
    
    @staticmethod
    def yearly_opens(df: pd.DataFrame) -> pd.Series:
        """
        Calculate yearly open prices
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Series with yearly opens
        """
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        
        yearly_opens = df_copy.groupby('year')['open'].first()
        
        # Forward fill
        result = pd.Series(index=df.index, dtype=float)
        for year, open_price in yearly_opens.items():
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year+1}-01-01')
            mask = (df.index >= year_start) & (df.index < year_end)
            result[mask] = open_price
        
        return result
    
    @staticmethod
    def session_levels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate session high/low (Overnight High/Low)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (ONH series, ONL series)
        """
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        # Group by date and get high/low
        onh = df_copy.groupby('date')['high'].max()
        onl = df_copy.groupby('date')['low'].min()
        
        # Map back to original index
        result_onh = df_copy['date'].map(onh)
        result_onl = df_copy['date'].map(onl)
        
        result_onh.index = df.index
        result_onl.index = df.index
        
        return result_onh, result_onl
    
    @staticmethod
    def market_profile(df: pd.DataFrame, value_area_pct: float = 0.70) -> Dict[str, pd.Series]:
        """
        Calculate Market Profile levels: POC, VAH, VAL
        
        Args:
            df: OHLCV DataFrame
            value_area_pct: Percentage for value area (default 70%)
            
        Returns:
            Dictionary with POC, VAH, VAL series
        """
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        poc_list = []
        vah_list = []
        val_list = []
        dates = []
        
        for date, group in df_copy.groupby('date'):
            # Create price bins
            price_range = group['high'].max() - group['low'].min()
            num_bins = 50
            bins = np.linspace(group['low'].min(), group['high'].max(), num_bins)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(num_bins - 1)
            
            for idx, row in group.iterrows():
                # Distribute volume across price range of candle
                low_idx = np.digitize(row['low'], bins) - 1
                high_idx = np.digitize(row['high'], bins) - 1
                
                if low_idx == high_idx:
                    volume_profile[low_idx] += row['volume']
                else:
                    volume_per_bin = row['volume'] / (high_idx - low_idx + 1)
                    volume_profile[low_idx:high_idx+1] += volume_per_bin
            
            # Point of Control (highest volume price)
            poc_idx = np.argmax(volume_profile)
            poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Value Area (70% of volume)
            sorted_indices = np.argsort(volume_profile)[::-1]
            total_volume = volume_profile.sum()
            target_volume = total_volume * value_area_pct
            
            cumulative = 0
            value_area_indices = []
            for idx in sorted_indices:
                cumulative += volume_profile[idx]
                value_area_indices.append(idx)
                if cumulative >= target_volume:
                    break
            
            # Value Area High/Low
            vah = bins[max(value_area_indices) + 1]
            val = bins[min(value_area_indices)]
            
            poc_list.append(poc)
            vah_list.append(vah)
            val_list.append(val)
            dates.append(date)
        
        # Create series and map to original index
        poc_series = pd.Series(poc_list, index=dates)
        vah_series = pd.Series(vah_list, index=dates)
        val_series = pd.Series(val_list, index=dates)
        
        result_poc = df_copy['date'].map(poc_series)
        result_vah = df_copy['date'].map(vah_series)
        result_val = df_copy['date'].map(val_series)
        
        result_poc.index = df.index
        result_vah.index = df.index
        result_val.index = df.index
        
        return {
            'poc': result_poc,
            'vah': result_vah,
            'val': result_val
        }
    
    @staticmethod
    def range_extremes(df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate multi-day/week range highs and lows
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of periods to look back
            
        Returns:
            Tuple of (range high, range low)
        """
        range_high = df['high'].rolling(window=lookback).max()
        range_low = df['low'].rolling(window=lookback).min()
        
        return range_high, range_low
    
    @staticmethod
    def composite_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        Calculate composite range levels (mid, quarters, thirds)
        
        Args:
            df: OHLCV DataFrame
            lookback: Period for range calculation
            
        Returns:
            Dictionary with composite levels
        """
        recent_data = df.tail(lookback)
        range_high = recent_data['high'].max()
        range_low = recent_data['low'].min()
        range_size = range_high - range_low
        
        return {
            'high': range_high,
            'low': range_low,
            'mid': range_low + (range_size * 0.5),
            'quarter_top': range_low + (range_size * 0.75),
            'quarter_bottom': range_low + (range_size * 0.25),
            'third_top': range_low + (range_size * 0.667),
            'third_mid': range_low + (range_size * 0.5),
            'third_bottom': range_low + (range_size * 0.333)
        }
    
    @staticmethod
    def identify_demand_zones(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Identify demand zones (support areas with strong buying)
        
        Args:
            df: OHLCV DataFrame with indicators
            lookback: Period to scan
            
        Returns:
            List of demand zone dictionaries
        """
        zones = []
        recent_data = df.tail(lookback)
        
        # Find local lows with volume spikes
        for i in range(2, len(recent_data) - 2):
            current = recent_data.iloc[i]
            
            # Check if it's a local low
            if (current['low'] < recent_data.iloc[i-1]['low'] and 
                current['low'] < recent_data.iloc[i+1]['low']):
                
                # Check for volume spike or high delta
                avg_volume = recent_data['volume'].mean()
                if current['volume'] > avg_volume * 1.5:
                    zones.append({
                        'timestamp': current.name,
                        'price_low': current['low'],
                        'price_high': current['low'] * 1.02,  # 2% zone
                        'volume': current['volume'],
                        'type': 'demand'
                    })
        
        return zones
    
    @staticmethod
    def identify_supply_zones(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Identify supply zones (resistance areas with strong selling)
        
        Args:
            df: OHLCV DataFrame with indicators
            lookback: Period to scan
            
        Returns:
            List of supply zone dictionaries
        """
        zones = []
        recent_data = df.tail(lookback)
        
        # Find local highs with volume spikes
        for i in range(2, len(recent_data) - 2):
            current = recent_data.iloc[i]
            
            # Check if it's a local high
            if (current['high'] > recent_data.iloc[i-1]['high'] and 
                current['high'] > recent_data.iloc[i+1]['high']):
                
                # Check for volume spike
                avg_volume = recent_data['volume'].mean()
                if current['volume'] > avg_volume * 1.5:
                    zones.append({
                        'timestamp': current.name,
                        'price_high': current['high'],
                        'price_low': current['high'] * 0.98,  # 2% zone
                        'volume': current['volume'],
                        'type': 'supply'
                    })
        
        return zones

