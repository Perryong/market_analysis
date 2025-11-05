"""Main signal generation engine using strategy pattern"""

from typing import List, Tuple
import numpy as np

from core.models import MarketData, TradingSignal, EntryZone
from core.enums import SignalType
from config.settings import ScoringConfig
from analysis.strategies.base import IndicatorStrategy


class SignalAnalyzer:
    """
    Main signal analysis engine that orchestrates multiple strategies
    using the Strategy Pattern for extensibility
    """
    
    def __init__(self, config: ScoringConfig):
        """
        Initialize analyzer with configuration
        
        Args:
            config: Scoring configuration with weights and thresholds
        """
        self.config = config
        self.strategies: List[IndicatorStrategy] = []
        
    def register_strategy(self, strategy: IndicatorStrategy):
        """
        Register a new indicator strategy
        
        Args:
            strategy: Strategy implementing IndicatorStrategy protocol
        """
        self.strategies.append(strategy)
        
    def analyze(self, market_data: MarketData) -> TradingSignal:
        """
        Generate comprehensive trading signal from market data
        
        Args:
            market_data: Encapsulated market data with indicators
            
        Returns:
            Complete trading signal with score, type, and reasons
        """
        # Execute all registered strategies
        scores: List[Tuple[float, float]] = []  # (weight, score)
        explanations: List[str] = []
        candle_score = 0.0
        
        for strategy in self.strategies:
            score, reason = strategy.evaluate(market_data)
            weight = strategy.weight
            
            scores.append((weight, score))
            explanations.append(reason)
            
            # Track candlestick score separately for edge case handling
            if "Candlestick" in strategy.name:
                candle_score = (score * 2) - 1  # Denormalize to -1...1
        
        # Calculate weighted average score
        total_weight = sum(w for w, _ in scores)
        if total_weight > 0:
            base_score = sum(w * s for w, s in scores) / total_weight
        else:
            base_score = 0.5
        
        # Apply adjustments
        adjusted_score = self._apply_adjustments(base_score, market_data, explanations)
        
        # Determine signal type
        signal_type = self._determine_signal(adjusted_score, candle_score)
        
        # Calculate entry zone
        entry_zone = self._calculate_entry_zone(market_data)
        
        return TradingSignal(
            ticker=market_data.ticker,
            short_name=market_data.short_name or market_data.ticker,
            signal=signal_type,
            confidence=adjusted_score,
            reasons=explanations,
            entry_zone=entry_zone,
            last_close=market_data.last_close,
            atr=market_data.atr,
            timeframe=market_data.timeframe
        )
    
    def _apply_adjustments(self, score: float, data: MarketData, 
                          explanations: List[str]) -> float:
        """
        Apply ADX and S/R proximity adjustments to base score
        
        Args:
            score: Base weighted score
            data: Market data with ADX and S/R values
            explanations: List to append adjustment reasons
            
        Returns:
            Adjusted score
        """
        # ADX adjustment (trend strength)
        adx_multiplier = self.config.adjustments.get_multiplier(data.adx)
        
        if adx_multiplier != 1.0:
            if data.adx < self.config.adjustments.weak_trend_threshold:
                explanations.append(f"Weak trend (ADX={data.adx:.1f}) - reduced confidence")
            elif data.adx > self.config.adjustments.strong_trend_threshold:
                explanations.append(f"Strong trend (ADX={data.adx:.1f}) - boosted confidence")
        
        # S/R proximity adjustment
        snr_multiplier = self.config.snr_config.get_multiplier(
            data.last_close, data.snr
        )
        
        if not np.isnan(data.snr) and snr_multiplier != 1.0:
            explanations.append(f"Near S/R level at ${data.snr:.2f}")
        
        # Apply multipliers
        adjusted = score * adx_multiplier * snr_multiplier
        
        # Clamp to valid range
        return min(1.0, max(0.0, adjusted))
    
    def _determine_signal(self, score: float, candle_score: float) -> SignalType:
        """
        Determine signal type from score and candlestick patterns
        
        Args:
            score: Adjusted weighted score (0.0 to 1.0)
            candle_score: Candlestick pattern score (-1.0 to 1.0)
            
        Returns:
            Signal type enum
        """
        thresholds = self.config.thresholds
        
        # Primary thresholds
        if score >= thresholds.buy_threshold:
            return SignalType.BUY
        elif score <= thresholds.sell_threshold:
            return SignalType.SELL
        else:
            # Edge case: strong candlestick pattern overrides
            if candle_score > thresholds.strong_candle_threshold:
                return SignalType.BUY
            elif candle_score < -thresholds.strong_candle_threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD
    
    def _calculate_entry_zone(self, data: MarketData) -> EntryZone:
        """
        Calculate suggested entry price range based on ATR
        
        Args:
            data: Market data with close price and ATR
            
        Returns:
            Entry zone with lower/upper bounds
        """
        return EntryZone.from_atr(
            close_price=data.last_close,
            atr=data.atr,
            lower_mult=self.config.entry_zone.lower_multiplier,
            upper_mult=self.config.entry_zone.upper_multiplier
        )

