"""Trend-based indicator strategies (Moving Averages)"""

from typing import Tuple
from core.models import MarketData


class MovingAverageCrossStrategy:
    """Moving Average Crossover trend strategy (MA50 vs MA200)"""
    
    def __init__(self, weight: float = 0.25):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "MA Crossover Trend"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate moving average crossover"""
        ma50 = float(data.current['ma50'])
        ma200 = float(data.current['ma200'])
        close = float(data.current['Close'])
        
        # Previous values for trend detection
        ma50_prev = float(data.previous['ma50'])
        ma200_prev = float(data.previous['ma200'])
        
        # Calculate spreads
        current_spread = ma50 - ma200
        previous_spread = ma50_prev - ma200_prev
        spread_pct = (current_spread / ma200) * 100 if ma200 > 0 else 0
        
        # Detect crossovers
        golden_cross = (previous_spread <= 0) and (current_spread > 0)
        death_cross = (previous_spread >= 0) and (current_spread < 0)
        
        # Price position relative to MAs
        above_both = close > ma50 and close > ma200
        below_both = close < ma50 and close < ma200
        
        # Scoring logic
        if golden_cross:
            score, label = 1.0, "Golden Cross (strong bullish)"
        elif ma50 > ma200:
            # Bullish configuration
            if above_both and spread_pct > 5:
                score, label = 0.9, "Strong Bullish Trend"
            elif above_both:
                score, label = 0.75, "Bullish Trend"
            else:
                score, label = 0.65, "Mild Bullish Trend"
        elif death_cross:
            score, label = 0.0, "Death Cross (strong bearish)"
        elif ma50 < ma200:
            # Bearish configuration
            if below_both and spread_pct < -5:
                score, label = 0.1, "Strong Bearish Trend"
            elif below_both:
                score, label = 0.25, "Bearish Trend"
            else:
                score, label = 0.35, "Mild Bearish Trend"
        else:
            # Near crossover or neutral
            score, label = 0.5, "Neutral"
        
        explanation = (
            f"MA50 {ma50:.2f} vs MA200 {ma200:.2f} - "
            f"{label} (spread: {spread_pct:.1f}%)"
        )
        
        return score, explanation

