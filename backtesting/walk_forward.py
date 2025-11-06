"""
Walk-Forward Analysis

Validates strategy robustness through rolling window optimization and testing.
Splits data into training/testing periods to detect overfitting.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.models import BacktestResult
from backtesting.metrics import MetricsCalculator, PerformanceMetrics
from analysis.signal_generator import SignalAnalyzer
from core.enums import TimeFrame


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result"""
    window_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Training results
    train_result: Optional[BacktestResult] = None
    train_metrics: Optional[PerformanceMetrics] = None
    
    # Testing results (out-of-sample)
    test_result: Optional[BacktestResult] = None
    test_metrics: Optional[PerformanceMetrics] = None
    
    # Optimal parameters found in training
    optimal_params: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'window_num': self.window_num,
            'train_start': self.train_start.strftime('%Y-%m-%d'),
            'train_end': self.train_end.strftime('%Y-%m-%d'),
            'test_start': self.test_start.strftime('%Y-%m-%d'),
            'test_end': self.test_end.strftime('%Y-%m-%d'),
            'train_return': self.train_metrics.total_return_percent if self.train_metrics else None,
            'train_sharpe': self.train_metrics.sharpe_ratio if self.train_metrics else None,
            'test_return': self.test_metrics.total_return_percent if self.test_metrics else None,
            'test_sharpe': self.test_metrics.sharpe_ratio if self.test_metrics else None,
            'optimal_params': self.optimal_params,
        }


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result"""
    ticker: str
    total_windows: int
    train_days: int
    test_days: int
    
    windows: List[WalkForwardWindow] = field(default_factory=list)
    
    # Aggregate metrics
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 0.0  # How similar train vs test performance
    win_rate_windows: float = 0.0  # % of windows that were profitable
    
    # Robustness indicators
    overfitting_detected: bool = False
    stability_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ticker': self.ticker,
            'total_windows': self.total_windows,
            'train_days': self.train_days,
            'test_days': self.test_days,
            'avg_train_return': self.avg_train_return,
            'avg_test_return': self.avg_test_return,
            'avg_train_sharpe': self.avg_train_sharpe,
            'avg_test_sharpe': self.avg_test_sharpe,
            'consistency_score': self.consistency_score,
            'win_rate_windows': self.win_rate_windows,
            'overfitting_detected': self.overfitting_detected,
            'stability_score': self.stability_score,
        }


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation
    
    Performs rolling window optimization and testing to validate
    strategy robustness and detect overfitting.
    """
    
    def __init__(self,
                 signal_analyzer: SignalAnalyzer,
                 config: Optional[BacktestConfig] = None):
        """
        Initialize walk-forward analyzer
        
        Args:
            signal_analyzer: Signal generator for strategy
            config: Backtesting configuration
        """
        self.signal_analyzer = signal_analyzer
        self.config = config or BacktestConfig()
        self.engine = BacktestEngine(signal_analyzer, config)
    
    def run(self,
            ticker: str,
            start_date: str,
            end_date: str,
            train_days: int = 365,
            test_days: int = 90,
            timeframe: TimeFrame = TimeFrame.DAILY) -> WalkForwardResult:
        """
        Run walk-forward analysis
        
        Args:
            ticker: Stock ticker symbol
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            train_days: Training window size in days
            test_days: Testing window size in days
            timeframe: Data timeframe
            
        Returns:
            WalkForwardResult with all windows
        """
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS: {ticker}")
        print(f"{'='*70}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Train Window: {train_days} days")
        print(f"Test Window: {test_days} days")
        print(f"{'='*70}")
        
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate windows
        windows = self._generate_windows(start_dt, end_dt, train_days, test_days)
        
        print(f"\n[*] Generated {len(windows)} walk-forward windows")
        
        # Process each window
        processed_windows = []
        for i, window in enumerate(windows):
            print(f"\n[WINDOW {i+1}/{len(windows)}]")
            print(f"  Train: {window.train_start.date()} to {window.train_end.date()}")
            print(f"  Test:  {window.test_start.date()} to {window.test_end.date()}")
            
            try:
                # Train phase
                print(f"  [*] Training...")
                window.train_result = self.engine.run(
                    ticker,
                    window.train_start.strftime('%Y-%m-%d'),
                    window.train_end.strftime('%Y-%m-%d'),
                    timeframe
                )
                window.train_metrics = MetricsCalculator.calculate(window.train_result)
                
                # Test phase (out-of-sample)
                print(f"  [*] Testing (out-of-sample)...")
                window.test_result = self.engine.run(
                    ticker,
                    window.test_start.strftime('%Y-%m-%d'),
                    window.test_end.strftime('%Y-%m-%d'),
                    timeframe
                )
                window.test_metrics = MetricsCalculator.calculate(window.test_result)
                
                # Print window summary
                print(f"  Train Return: {window.train_metrics.total_return_percent:.2f}% | "
                      f"Sharpe: {window.train_metrics.sharpe_ratio:.2f}")
                print(f"  Test Return:  {window.test_metrics.total_return_percent:.2f}% | "
                      f"Sharpe: {window.test_metrics.sharpe_ratio:.2f}")
                
                processed_windows.append(window)
                
            except Exception as e:
                print(f"  [ERROR] Window failed: {e}")
                continue
        
        # Calculate aggregate statistics
        wf_result = self._calculate_aggregate_stats(
            ticker, processed_windows, train_days, test_days
        )
        
        self._print_summary(wf_result)
        
        return wf_result
    
    def _generate_windows(self,
                         start_date: datetime,
                         end_date: datetime,
                         train_days: int,
                         test_days: int) -> List[WalkForwardWindow]:
        """Generate walk-forward windows"""
        windows = []
        window_num = 1
        current_date = start_date
        
        while current_date + timedelta(days=train_days + test_days) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)
            
            window = WalkForwardWindow(
                window_num=window_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            
            # Move forward by test_days (anchored walk-forward)
            current_date = test_start
            window_num += 1
        
        return windows
    
    def _calculate_aggregate_stats(self,
                                   ticker: str,
                                   windows: List[WalkForwardWindow],
                                   train_days: int,
                                   test_days: int) -> WalkForwardResult:
        """Calculate aggregate statistics across all windows"""
        
        wf_result = WalkForwardResult(
            ticker=ticker,
            total_windows=len(windows),
            train_days=train_days,
            test_days=test_days,
            windows=windows
        )
        
        if len(windows) == 0:
            return wf_result
        
        # Average returns
        train_returns = [w.train_metrics.total_return_percent 
                        for w in windows if w.train_metrics]
        test_returns = [w.test_metrics.total_return_percent 
                       for w in windows if w.test_metrics]
        
        if train_returns:
            wf_result.avg_train_return = np.mean(train_returns)
        if test_returns:
            wf_result.avg_test_return = np.mean(test_returns)
        
        # Average Sharpe ratios
        train_sharpes = [w.train_metrics.sharpe_ratio 
                        for w in windows if w.train_metrics]
        test_sharpes = [w.test_metrics.sharpe_ratio 
                       for w in windows if w.test_metrics]
        
        if train_sharpes:
            wf_result.avg_train_sharpe = np.mean(train_sharpes)
        if test_sharpes:
            wf_result.avg_test_sharpe = np.mean(test_sharpes)
        
        # Consistency score (correlation between train and test returns)
        if len(train_returns) > 1 and len(test_returns) > 1 and len(train_returns) == len(test_returns):
            correlation = np.corrcoef(train_returns, test_returns)[0, 1]
            wf_result.consistency_score = max(0, correlation)  # Clip to [0, 1]
        
        # Win rate (percentage of profitable test windows)
        profitable_windows = sum(1 for r in test_returns if r > 0)
        wf_result.win_rate_windows = (profitable_windows / len(test_returns)) * 100 if test_returns else 0
        
        # Overfitting detection
        # If train performance significantly better than test, likely overfitting
        if wf_result.avg_train_return > wf_result.avg_test_return * 1.5:
            wf_result.overfitting_detected = True
        
        # Stability score (inverse of std deviation in test returns)
        if test_returns and len(test_returns) > 1:
            test_std = np.std(test_returns)
            # Normalize to [0, 1] where higher is more stable
            wf_result.stability_score = 1.0 / (1.0 + test_std / 10.0)
        
        return wf_result
    
    def _print_summary(self, result: WalkForwardResult):
        """Print walk-forward analysis summary"""
        print(f"\n{'='*70}")
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nWindows Analyzed: {result.total_windows}")
        print(f"Training Period: {result.train_days} days")
        print(f"Testing Period: {result.test_days} days")
        
        print("\n--- AVERAGE PERFORMANCE ---")
        print(f"Training Return:  {result.avg_train_return:.2f}%")
        print(f"Testing Return:   {result.avg_test_return:.2f}%")
        print(f"Training Sharpe:  {result.avg_train_sharpe:.3f}")
        print(f"Testing Sharpe:   {result.avg_test_sharpe:.3f}")
        
        print("\n--- ROBUSTNESS METRICS ---")
        print(f"Consistency Score:     {result.consistency_score:.3f}")
        print(f"Win Rate (Windows):    {result.win_rate_windows:.1f}%")
        print(f"Stability Score:       {result.stability_score:.3f}")
        
        if result.overfitting_detected:
            print(f"\n[WARNING] Overfitting detected! Train performance >> Test performance")
            print("  Consider: simplifying strategy, adding regularization, or more data")
        else:
            print(f"\n[OK] No significant overfitting detected")
        
        print(f"{'='*70}\n")
    
    def export_windows_csv(self, result: WalkForwardResult, filepath: str):
        """Export window-by-window results to CSV"""
        window_data = [w.to_dict() for w in result.windows]
        df = pd.DataFrame(window_data)
        df.to_csv(filepath, index=False)
        print(f"[*] Exported walk-forward windows to {filepath}")
    
    def plot_walk_forward_results(self, result: WalkForwardResult):
        """Create visualization of walk-forward results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Train vs Test Returns
            window_nums = [w.window_num for w in result.windows]
            train_returns = [w.train_metrics.total_return_percent if w.train_metrics else 0 
                           for w in result.windows]
            test_returns = [w.test_metrics.total_return_percent if w.test_metrics else 0 
                          for w in result.windows]
            
            ax1.plot(window_nums, train_returns, 'o-', label='Train', linewidth=2)
            ax1.plot(window_nums, test_returns, 's-', label='Test (Out-of-Sample)', linewidth=2)
            ax1.axhline(0, color='black', linestyle='--', linewidth=1)
            ax1.set_title('Returns by Window', fontweight='bold')
            ax1.set_xlabel('Window Number')
            ax1.set_ylabel('Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Train vs Test Sharpe
            train_sharpes = [w.train_metrics.sharpe_ratio if w.train_metrics else 0 
                           for w in result.windows]
            test_sharpes = [w.test_metrics.sharpe_ratio if w.test_metrics else 0 
                          for w in result.windows]
            
            ax2.plot(window_nums, train_sharpes, 'o-', label='Train', linewidth=2)
            ax2.plot(window_nums, test_sharpes, 's-', label='Test (Out-of-Sample)', linewidth=2)
            ax2.axhline(0, color='black', linestyle='--', linewidth=1)
            ax2.set_title('Sharpe Ratio by Window', fontweight='bold')
            ax2.set_xlabel('Window Number')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Scatter: Train vs Test Returns
            ax3.scatter(train_returns, test_returns, alpha=0.6, s=100)
            
            # Add diagonal line (perfect correlation)
            min_val = min(min(train_returns), min(test_returns))
            max_val = max(max(train_returns), max(test_returns))
            ax3.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Correlation')
            
            ax3.set_title('Train vs Test Returns', fontweight='bold')
            ax3.set_xlabel('Train Return (%)')
            ax3.set_ylabel('Test Return (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics bar chart
            categories = ['Avg Return\n(Train)', 'Avg Return\n(Test)', 
                         'Win Rate\n(Windows)', 'Consistency\nScore']
            values = [result.avg_train_return, result.avg_test_return,
                     result.win_rate_windows, result.consistency_score * 100]
            colors = ['blue', 'green', 'orange', 'purple']
            
            ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_title('Summary Statistics', fontweight='bold')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save plot
            from pathlib import Path
            output_dir = Path("results/backtesting")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{result.ticker}_walk_forward.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[*] Saved walk-forward plot to {filepath}")
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Could not create plot: {e}")

