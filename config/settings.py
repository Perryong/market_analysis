"""Configuration settings for trading signal system"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import yaml
from pathlib import Path


@dataclass
class IndicatorWeights:
    """Scoring weights configuration for technical indicators"""
    macd: float = 0.25
    ma_crossover: float = 0.25
    rsi: float = 0.10
    obv: float = 0.10
    bollinger_bands: float = 0.10
    volume: float = 0.10
    candlestick: float = 0.10
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)
    
    @property
    def total_weight(self) -> float:
        """Calculate total weight (should be ~1.0)"""
        return sum(self.as_dict().values())


@dataclass
class SignalThresholds:
    """Signal decision thresholds"""
    buy_threshold: float = 0.65
    sell_threshold: float = 0.15
    strong_candle_threshold: float = 0.70
    
    def classify_score(self, score: float, candle_score: float = 0.0) -> str:
        """Classify score into signal type"""
        from core.enums import SignalType
        
        if score >= self.buy_threshold:
            return SignalType.BUY
        elif score <= self.sell_threshold:
            return SignalType.SELL
        else:
            # Check candlestick patterns for edge cases
            if candle_score > self.strong_candle_threshold:
                return SignalType.BUY
            elif candle_score < -self.strong_candle_threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD


@dataclass
class TrendAdjustments:
    """ADX-based trend strength adjustments"""
    weak_trend_threshold: int = 20
    strong_trend_threshold: int = 40
    weak_multiplier: float = 0.8
    neutral_multiplier: float = 1.0
    strong_multiplier: float = 1.2
    
    def get_multiplier(self, adx: float) -> float:
        """Get multiplier based on ADX value"""
        if adx < self.weak_trend_threshold:
            return self.weak_multiplier
        elif adx > self.strong_trend_threshold:
            return self.strong_multiplier
        else:
            return self.neutral_multiplier


@dataclass
class SupportResistanceConfig:
    """Support/Resistance proximity configuration"""
    tolerance_percent: float = 0.05  # 5% tolerance
    near_multiplier: float = 1.2
    far_multiplier: float = 0.8
    proximity_threshold: float = 0.7  # Strength threshold for "near"
    
    def get_multiplier(self, close_price: float, snr_level: float) -> float:
        """Calculate multiplier based on proximity to S/R level"""
        import numpy as np
        
        if np.isnan(snr_level):
            return 1.0
        
        tolerance = close_price * self.tolerance_percent
        distance = abs(close_price - snr_level)
        proximity_strength = max(0, 1 - distance / tolerance)
        
        if proximity_strength > self.proximity_threshold:
            return self.near_multiplier
        else:
            return self.far_multiplier


@dataclass
class EntryZoneConfig:
    """Entry zone calculation configuration"""
    lower_multiplier: float = 0.5  # ATR multiplier for lower bound
    upper_multiplier: float = 0.8  # ATR multiplier for upper bound


@dataclass
class BacktestingConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    position_size: float = 0.1  # 10% per position
    max_positions: int = 10
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    holding_period_days: Optional[int] = None


@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration"""
    train_days: int = 365
    test_days: int = 90


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration"""
    num_simulations: int = 1000
    confidence_levels: list = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]


@dataclass
class ScoringConfig:
    """Complete scoring configuration"""
    weights: IndicatorWeights = None
    thresholds: SignalThresholds = None
    adjustments: TrendAdjustments = None
    snr_config: SupportResistanceConfig = None
    entry_zone: EntryZoneConfig = None
    backtesting: BacktestingConfig = None
    walk_forward: WalkForwardConfig = None
    monte_carlo: MonteCarloConfig = None
    
    def __post_init__(self):
        """Initialize with defaults if not provided"""
        if self.weights is None:
            self.weights = IndicatorWeights()
        if self.thresholds is None:
            self.thresholds = SignalThresholds()
        if self.adjustments is None:
            self.adjustments = TrendAdjustments()
        if self.snr_config is None:
            self.snr_config = SupportResistanceConfig()
        if self.entry_zone is None:
            self.entry_zone = EntryZoneConfig()
        if self.backtesting is None:
            self.backtesting = BacktestingConfig()
        if self.walk_forward is None:
            self.walk_forward = WalkForwardConfig()
        if self.monte_carlo is None:
            self.monte_carlo = MonteCarloConfig()
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ScoringConfig':
        """Load configuration from YAML file"""
        path = Path(filepath)
        if not path.exists():
            print(f"[WARNING] Config file not found: {filepath}, using defaults")
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            weights=IndicatorWeights(**data.get('weights', {})),
            thresholds=SignalThresholds(**data.get('thresholds', {})),
            adjustments=TrendAdjustments(**data.get('adjustments', {})),
            snr_config=SupportResistanceConfig(**data.get('snr_config', {})),
            entry_zone=EntryZoneConfig(**data.get('entry_zone', {})),
            backtesting=BacktestingConfig(**data.get('backtesting', {})),
            walk_forward=WalkForwardConfig(**data.get('walk_forward', {})),
            monte_carlo=MonteCarloConfig(**data.get('monte_carlo', {}))
        )
    
    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        data = {
            'weights': asdict(self.weights),
            'thresholds': asdict(self.thresholds),
            'adjustments': asdict(self.adjustments),
            'snr_config': asdict(self.snr_config),
            'entry_zone': asdict(self.entry_zone),
            'backtesting': asdict(self.backtesting),
            'walk_forward': asdict(self.walk_forward),
            'monte_carlo': asdict(self.monte_carlo)
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

