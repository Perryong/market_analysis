"""Momentum-based indicator strategies (RSI, MACD, OBV)"""

from typing import Tuple
from core.models import MarketData


class RSIStrategy:
    """RSI (Relative Strength Index) momentum strategy"""
    
    def __init__(self, weight: float = 0.10):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "RSI Momentum"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate RSI indicator"""
        rsi_current = float(data.current['rsi'])
        rsi_previous = float(data.previous['rsi'])
        
        # Determine trend direction
        trending_up = rsi_previous < rsi_current
        trending_down = rsi_previous > rsi_current
        
        # Score based on RSI zones and trend
        if rsi_current < 30:
            # Oversold zone
            score = 1.0 if trending_up else 0.8
            sentiment = "Strong Bullish" if trending_up else "Bullish"
        elif rsi_current < 40:
            # Mildly oversold
            score = 0.7 if trending_up else 0.5
            sentiment = "Bullish" if trending_up else "Neutral"
        elif rsi_current < 60:
            # Neutral zone
            score = 0.5
            sentiment = "Neutral"
        elif rsi_current < 70:
            # Mildly overbought
            score = 0.3 if trending_down else 0.5
            sentiment = "Bearish" if trending_down else "Neutral"
        else:
            # Overbought zone
            score = 0.2 if trending_down else 0.0
            sentiment = "Bearish" if trending_down else "Strong Bearish"
        
        # Trend indicator
        trend = '^' if trending_up else 'v' if trending_down else '-'
        
        explanation = f"RSI {rsi_current:.1f} - {sentiment} (trend: {trend})"
        
        return score, explanation


class MACDStrategy:
    """MACD (Moving Average Convergence Divergence) trend/momentum strategy"""
    
    def __init__(self, weight: float = 0.25):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "MACD Trend/Momentum"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate MACD indicator"""
        macd = float(data.current['macd'])
        signal = float(data.current['macd_signal'])
        histogram = float(data.current['macd_hist'])
        
        macd_prev = float(data.previous['macd'])
        signal_prev = float(data.previous['macd_signal'])
        hist_prev = float(data.previous['macd_hist'])
        
        # Calculate spreads and changes
        spread = macd - signal
        spread_prev = macd_prev - signal_prev
        spread_delta = spread - spread_prev
        hist_change = histogram - hist_prev
        
        # Detect crossovers
        bullish_cross = (spread_prev <= 0) and (spread > 0)
        bearish_cross = (spread_prev >= 0) and (spread < 0)
        
        # Scoring logic based on multiple conditions
        if bullish_cross and histogram > 0 and hist_change > 0:
            score, label = 1.0, "Strong Bullish Crossover"
        elif spread > 0 and spread_delta > 0 and histogram > 0:
            score, label = 0.8, "Bullish Momentum"
        elif spread > 0 and histogram > 0:
            score, label = 0.65, "Mild Bullish"
        elif bearish_cross and histogram < 0 and hist_change < 0:
            score, label = 0.0, "Strong Bearish Crossover"
        elif spread < 0 and spread_delta < 0 and histogram < 0:
            score, label = 0.2, "Bearish Momentum"
        else:
            # Neutral with histogram bias
            score = 0.55 if histogram > 0 else 0.45
            label = "Neutral Bullish" if histogram > 0 else "Neutral Bearish"
        
        explanation = (
            f"MACD {label} "
            f"(spread: {spread:.4f}, d_hist: {hist_change:.4f})"
        )
        
        return score, explanation


class OBVStrategy:
    """OBV (On-Balance Volume) volume flow strategy"""
    
    def __init__(self, weight: float = 0.10):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "OBV Volume Flow"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate OBV slope indicator"""
        obv_slope = float(data.current['obv_slope'])
        obv_prev = float(data.previous['obv_slope'])
        
        # Tolerance for near-zero values
        epsilon = 1e-9
        
        if abs(obv_slope) <= epsilon:
            score, label = 0.5, "Neutral"
        elif obv_slope > 0:
            # Positive slope = accumulation (buying pressure)
            if obv_slope > obv_prev:
                score, label = 1.0, "Strong Bullish"
            else:
                score, label = 0.8, "Bullish"
        else:
            # Negative slope = distribution (selling pressure)
            if obv_slope < obv_prev:
                score, label = 0.0, "Strong Bearish"
            else:
                score, label = 0.2, "Bearish"
        
        explanation = f"OBV slope {obv_slope:.1f} - {label} volume flow"
        
        return score, explanation

