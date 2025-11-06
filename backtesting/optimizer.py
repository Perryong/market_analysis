"""
Strategy Parameter Optimization

Grid search optimization for strategy parameters with overfitting detection
and comprehensive visualization of optimization results.
"""

from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass, field
import copy

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.metrics import MetricsCalculator, PerformanceMetrics
from analysis.signal_generator import SignalAnalyzer
from config.settings import ScoringConfig
from core.enums import TimeFrame


@dataclass
class OptimizationResult:
    """Single optimization run result"""
    params: Dict[str, Any]
    metric_value: float
    total_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    win_rate: float
    num_trades: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {'metric_value': self.metric_value}
        result.update(self.params)
        result.update({
            'total_return': self.total_return_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown_percent,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
        })
        return result


@dataclass
class GridSearchResult:
    """Complete grid search results"""
    ticker: str
    optimization_metric: str
    
    results: List[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    
    total_combinations: int = 0
    completed_combinations: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)
    
    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """Get top N results by metric value"""
        sorted_results = sorted(self.results, 
                               key=lambda x: x.metric_value, 
                               reverse=True)
        return sorted_results[:n]


class StrategyOptimizer:
    """
    Parameter optimization using grid search
    
    Systematically tests parameter combinations to find optimal values
    for maximizing a specific performance metric.
    """
    
    VALID_METRICS = [
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'total_return_percent',
        'profit_factor',
        'win_rate',
    ]
    
    def __init__(self,
                 base_config: ScoringConfig,
                 backtest_config: Optional[BacktestConfig] = None):
        """
        Initialize optimizer
        
        Args:
            base_config: Base strategy configuration
            backtest_config: Backtesting configuration
        """
        self.base_config = base_config
        self.backtest_config = backtest_config or BacktestConfig()
    
    def optimize(self,
                 ticker: str,
                 start_date: str,
                 end_date: str,
                 param_grid: Dict[str, List[Any]],
                 metric: str = 'sharpe_ratio',
                 timeframe: TimeFrame = TimeFrame.DAILY,
                 max_combinations: Optional[int] = None) -> GridSearchResult:
        """
        Run grid search optimization
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            param_grid: Dictionary of parameters and their values to test
                       e.g., {'buy_threshold': [0.6, 0.65, 0.7],
                              'sell_threshold': [0.1, 0.15, 0.2]}
            metric: Metric to optimize (see VALID_METRICS)
            timeframe: Data timeframe
            max_combinations: Maximum combinations to test (None = all)
            
        Returns:
            GridSearchResult with all tested combinations
        """
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric. Choose from: {self.VALID_METRICS}")
        
        print(f"\n{'='*70}")
        print(f"PARAMETER OPTIMIZATION: {ticker}")
        print(f"{'='*70}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Optimization Metric: {metric}")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        if max_combinations and total_combinations > max_combinations:
            print(f"\n[WARNING] Too many combinations ({total_combinations}). "
                  f"Limiting to {max_combinations}")
            # Random sample
            import random
            combinations = random.sample(combinations, max_combinations)
            total_combinations = max_combinations
        
        print(f"\nTotal Combinations: {total_combinations}")
        print(f"{'='*70}")
        
        # Test each combination
        grid_result = GridSearchResult(
            ticker=ticker,
            optimization_metric=metric,
            total_combinations=total_combinations
        )
        
        best_metric_value = float('-inf')
        
        for i, combo in enumerate(combinations):
            # Create parameter dict
            params = dict(zip(param_names, combo))
            
            print(f"\n[{i+1}/{total_combinations}] Testing: {params}")
            
            try:
                # Create config with these parameters
                config = self._create_config_with_params(params)
                
                # Run backtest
                signal_analyzer = SignalAnalyzer(config)
                self._register_strategies(signal_analyzer, config)
                
                engine = BacktestEngine(signal_analyzer, self.backtest_config)
                backtest_result = engine.run(ticker, start_date, end_date, timeframe)
                
                # Calculate metrics
                metrics = MetricsCalculator.calculate(backtest_result)
                
                # Get metric value
                metric_value = self._get_metric_value(metrics, metric)
                
                # Store result
                opt_result = OptimizationResult(
                    params=params,
                    metric_value=metric_value,
                    total_return_percent=metrics.total_return_percent,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown_percent=metrics.max_drawdown_percent,
                    win_rate=metrics.win_rate,
                    num_trades=metrics.num_trades
                )
                
                grid_result.results.append(opt_result)
                grid_result.completed_combinations += 1
                
                # Track best
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    grid_result.best_result = opt_result
                    grid_result.best_params = params
                    print(f"  [NEW BEST] {metric} = {metric_value:.4f}")
                else:
                    print(f"  {metric} = {metric_value:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue
        
        # Print summary
        self._print_summary(grid_result)
        
        return grid_result
    
    def _create_config_with_params(self, params: Dict[str, Any]) -> ScoringConfig:
        """Create new config with specified parameters"""
        config = copy.deepcopy(self.base_config)
        
        # Update config based on parameter names
        for param_name, param_value in params.items():
            if hasattr(config.thresholds, param_name):
                setattr(config.thresholds, param_name, param_value)
            elif hasattr(config.weights, param_name):
                setattr(config.weights, param_name, param_value)
            elif hasattr(config.adjustments, param_name):
                setattr(config.adjustments, param_name, param_value)
        
        return config
    
    def _register_strategies(self, analyzer: SignalAnalyzer, config: ScoringConfig):
        """Register strategies with analyzer"""
        from analysis.strategies.momentum import RSIStrategy, MACDStrategy, OBVStrategy
        from analysis.strategies.trend import MovingAverageCrossStrategy
        from analysis.strategies.volatility import BollingerBandStrategy
        from analysis.strategies.volume import VolumeStrategy
        from technical.patterns.strategy import CandlestickPatternStrategy
        
        weights = config.weights
        
        analyzer.register_strategy(RSIStrategy(weights.rsi))
        analyzer.register_strategy(MACDStrategy(weights.macd))
        analyzer.register_strategy(OBVStrategy(weights.obv))
        analyzer.register_strategy(MovingAverageCrossStrategy(weights.ma_crossover))
        analyzer.register_strategy(BollingerBandStrategy(weights.bollinger_bands))
        analyzer.register_strategy(VolumeStrategy(weights.volume))
        analyzer.register_strategy(CandlestickPatternStrategy(weights.candlestick))
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric_name: str) -> float:
        """Extract metric value from metrics object"""
        return getattr(metrics, metric_name, 0.0)
    
    def _print_summary(self, result: GridSearchResult):
        """Print optimization summary"""
        print(f"\n{'='*70}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nCompleted: {result.completed_combinations}/{result.total_combinations} combinations")
        
        if result.best_result:
            print(f"\n--- BEST PARAMETERS ---")
            for param, value in result.best_params.items():
                print(f"{param}: {value}")
            
            print(f"\n--- BEST PERFORMANCE ---")
            print(f"{result.optimization_metric}: {result.best_result.metric_value:.4f}")
            print(f"Total Return: {result.best_result.total_return_percent:.2f}%")
            print(f"Sharpe Ratio: {result.best_result.sharpe_ratio:.3f}")
            print(f"Max Drawdown: {result.best_result.max_drawdown_percent:.2f}%")
            print(f"Win Rate: {result.best_result.win_rate:.1f}%")
            print(f"Num Trades: {result.best_result.num_trades}")
        
        # Show top 5
        print(f"\n--- TOP 5 PARAMETER SETS ---")
        top_5 = result.get_top_n(5)
        for i, opt_result in enumerate(top_5, 1):
            print(f"\n{i}. {result.optimization_metric}={opt_result.metric_value:.4f} | "
                  f"Return={opt_result.total_return_percent:.2f}% | "
                  f"Sharpe={opt_result.sharpe_ratio:.2f}")
            print(f"   Params: {opt_result.params}")
        
        print(f"\n{'='*70}\n")
    
    def plot_optimization_surface(self,
                                  result: GridSearchResult,
                                  param_x: str,
                                  param_y: str,
                                  save_path: Optional[str] = None):
        """
        Plot 2D optimization surface
        
        Args:
            result: Grid search result
            param_x: Parameter for X axis
            param_y: Parameter for Y axis
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            # Convert to DataFrame
            df = result.to_dataframe()
            
            # Filter to only rows with both parameters
            if param_x not in df.columns or param_y not in df.columns:
                print(f"[ERROR] Parameters {param_x} or {param_y} not in results")
                return
            
            # Create pivot table
            pivot = df.pivot_table(
                values='metric_value',
                index=param_y,
                columns=param_x,
                aggfunc='mean'
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
            
            # Create meshgrid
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            Z = pivot.values
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            # Add scatter for actual points
            ax.scatter(df[param_x], df[param_y], df['metric_value'],
                      c='red', marker='o', s=50, alpha=0.6)
            
            # Labels
            ax.set_xlabel(param_x, fontweight='bold', labelpad=10)
            ax.set_ylabel(param_y, fontweight='bold', labelpad=10)
            ax.set_zlabel(result.optimization_metric, fontweight='bold', labelpad=10)
            ax.set_title(f'Optimization Surface: {result.ticker}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[*] Saved optimization surface to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Could not create optimization surface: {e}")
    
    def export_results_csv(self, result: GridSearchResult, filepath: str):
        """Export optimization results to CSV"""
        df = result.to_dataframe()
        df = df.sort_values('metric_value', ascending=False)
        df.to_csv(filepath, index=False)
        print(f"[*] Exported optimization results to {filepath}")

