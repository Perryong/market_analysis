"""
Monte Carlo Simulation for Backtesting

Risk assessment through resampling trade results and generating
multiple equity curve scenarios to estimate confidence intervals
and probability distributions.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from backtesting.models import BacktestResult, Trade


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    
    # Simulation parameters
    num_simulations: int
    num_trades: int
    initial_capital: float
    
    # Simulated equity curves
    equity_curves: List[pd.Series] = field(default_factory=list)
    
    # Percentile curves
    percentile_5: Optional[pd.Series] = None
    percentile_25: Optional[pd.Series] = None
    percentile_50: Optional[pd.Series] = None
    percentile_75: Optional[pd.Series] = None
    percentile_95: Optional[pd.Series] = None
    
    # Risk metrics
    prob_profit: float = 0.0
    prob_loss: float = 0.0
    prob_ruin: float = 0.0
    expected_return: float = 0.0
    expected_return_percent: float = 0.0
    
    # Drawdown statistics
    avg_max_drawdown: float = 0.0
    worst_max_drawdown: float = 0.0
    prob_dd_over_20: float = 0.0
    prob_dd_over_30: float = 0.0
    
    # Final equity statistics
    final_equity_mean: float = 0.0
    final_equity_std: float = 0.0
    final_equity_min: float = 0.0
    final_equity_max: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'num_simulations': self.num_simulations,
            'num_trades': self.num_trades,
            'initial_capital': self.initial_capital,
            'prob_profit': self.prob_profit,
            'prob_loss': self.prob_loss,
            'prob_ruin': self.prob_ruin,
            'expected_return': self.expected_return,
            'expected_return_percent': self.expected_return_percent,
            'avg_max_drawdown': self.avg_max_drawdown,
            'worst_max_drawdown': self.worst_max_drawdown,
            'prob_dd_over_20': self.prob_dd_over_20,
            'prob_dd_over_30': self.prob_dd_over_30,
            'final_equity_mean': self.final_equity_mean,
            'final_equity_std': self.final_equity_std,
            'final_equity_min': self.final_equity_min,
            'final_equity_max': self.final_equity_max,
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for risk assessment
    
    Resamples trade returns randomly to generate multiple possible
    equity curve scenarios and calculate risk statistics.
    """
    
    def __init__(self, num_simulations: int = 1000, confidence_levels: List[float] = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            num_simulations: Number of simulation runs
            confidence_levels: Confidence levels for percentiles (default: [0.05, 0.5, 0.95])
        """
        self.num_simulations = num_simulations
        self.confidence_levels = confidence_levels or [0.05, 0.25, 0.50, 0.75, 0.95]
        
    def run(self, result: BacktestResult) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on backtest results
        
        Args:
            result: BacktestResult to simulate
            
        Returns:
            MonteCarloResult with simulation statistics
        """
        print(f"\n{'='*70}")
        print("MONTE CARLO SIMULATION")
        print(f"{'='*70}")
        print(f"Simulations: {self.num_simulations}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        
        closed_trades = result.closed_trades
        
        if len(closed_trades) == 0:
            print("[ERROR] No closed trades to simulate")
            return self._create_empty_result(result)
        
        print(f"Trades to resample: {len(closed_trades)}")
        
        # Extract trade returns
        trade_returns = self._extract_trade_returns(closed_trades)
        
        if len(trade_returns) == 0:
            print("[ERROR] No valid trade returns")
            return self._create_empty_result(result)
        
        # Run simulations
        print(f"[*] Running {self.num_simulations} simulations...")
        
        equity_curves = []
        final_equities = []
        max_drawdowns = []
        
        for i in range(self.num_simulations):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{self.num_simulations}")
            
            # Resample trades with replacement
            resampled_returns = np.random.choice(trade_returns, 
                                                size=len(trade_returns), 
                                                replace=True)
            
            # Generate equity curve
            equity_curve = self._simulate_equity_curve(
                resampled_returns, 
                result.initial_capital
            )
            
            equity_curves.append(equity_curve)
            final_equities.append(equity_curve.iloc[-1])
            
            # Calculate max drawdown for this simulation
            max_dd = self._calculate_max_drawdown_percent(equity_curve)
            max_drawdowns.append(max_dd)
        
        print(f"[SUCCESS] Completed {self.num_simulations} simulations")
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles(equity_curves)
        
        # Calculate statistics
        mc_result = self._calculate_statistics(
            result, equity_curves, final_equities, max_drawdowns, percentiles
        )
        
        self._print_summary(mc_result)
        
        return mc_result
    
    def _extract_trade_returns(self, trades: List[Trade]) -> np.ndarray:
        """Extract trade returns as percentage"""
        returns = []
        for trade in trades:
            if trade.pnl_percent is not None:
                returns.append(trade.pnl_percent / 100)  # Convert to decimal
        return np.array(returns)
    
    def _simulate_equity_curve(self, returns: np.ndarray, initial_capital: float) -> pd.Series:
        """Generate equity curve from resampled returns"""
        equity = [initial_capital]
        
        for ret in returns:
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)
        
        return pd.Series(equity)
    
    def _calculate_max_drawdown_percent(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage for an equity curve"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        return abs(drawdown.min())
    
    def _calculate_percentiles(self, equity_curves: List[pd.Series]) -> Dict[int, pd.Series]:
        """Calculate percentile curves"""
        # Convert list of series to DataFrame
        df = pd.DataFrame({i: curve.values for i, curve in enumerate(equity_curves)})
        
        percentiles = {}
        for conf in self.confidence_levels:
            pct_value = int(conf * 100)
            percentiles[pct_value] = df.quantile(conf, axis=1)
        
        return percentiles
    
    def _calculate_statistics(self,
                             result: BacktestResult,
                             equity_curves: List[pd.Series],
                             final_equities: List[float],
                             max_drawdowns: List[float],
                             percentiles: Dict[int, pd.Series]) -> MonteCarloResult:
        """Calculate all Monte Carlo statistics"""
        
        final_equities_arr = np.array(final_equities)
        max_drawdowns_arr = np.array(max_drawdowns)
        
        mc_result = MonteCarloResult(
            num_simulations=self.num_simulations,
            num_trades=len(result.closed_trades),
            initial_capital=result.initial_capital,
            equity_curves=equity_curves,
        )
        
        # Set percentile curves
        mc_result.percentile_5 = percentiles.get(5)
        mc_result.percentile_25 = percentiles.get(25)
        mc_result.percentile_50 = percentiles.get(50)
        mc_result.percentile_75 = percentiles.get(75)
        mc_result.percentile_95 = percentiles.get(95)
        
        # Probability metrics
        mc_result.prob_profit = np.mean(final_equities_arr > result.initial_capital)
        mc_result.prob_loss = np.mean(final_equities_arr < result.initial_capital)
        
        # Risk of ruin (losing >50% of capital)
        ruin_threshold = result.initial_capital * 0.5
        mc_result.prob_ruin = np.mean(final_equities_arr < ruin_threshold)
        
        # Expected return
        mc_result.final_equity_mean = float(final_equities_arr.mean())
        mc_result.expected_return = mc_result.final_equity_mean - result.initial_capital
        mc_result.expected_return_percent = (mc_result.expected_return / result.initial_capital) * 100
        
        # Final equity statistics
        mc_result.final_equity_std = float(final_equities_arr.std())
        mc_result.final_equity_min = float(final_equities_arr.min())
        mc_result.final_equity_max = float(final_equities_arr.max())
        
        # Drawdown statistics
        mc_result.avg_max_drawdown = float(max_drawdowns_arr.mean())
        mc_result.worst_max_drawdown = float(max_drawdowns_arr.max())
        mc_result.prob_dd_over_20 = np.mean(max_drawdowns_arr > 20)
        mc_result.prob_dd_over_30 = np.mean(max_drawdowns_arr > 30)
        
        return mc_result
    
    def _create_empty_result(self, result: BacktestResult) -> MonteCarloResult:
        """Create empty result for failed simulation"""
        return MonteCarloResult(
            num_simulations=0,
            num_trades=0,
            initial_capital=result.initial_capital
        )
    
    def _print_summary(self, result: MonteCarloResult):
        """Print Monte Carlo summary"""
        print(f"\n{'='*70}")
        print("MONTE CARLO RESULTS")
        print(f"{'='*70}")
        
        print("\n--- PROBABILITY METRICS ---")
        print(f"Probability of Profit:    {result.prob_profit*100:.1f}%")
        print(f"Probability of Loss:      {result.prob_loss*100:.1f}%")
        print(f"Risk of Ruin (>50% loss): {result.prob_ruin*100:.1f}%")
        
        print("\n--- EXPECTED RETURNS ---")
        print(f"Expected Final Equity:    ${result.final_equity_mean:,.2f}")
        print(f"Expected Return:          ${result.expected_return:,.2f} ({result.expected_return_percent:.2f}%)")
        print(f"Std Deviation:            ${result.final_equity_std:,.2f}")
        print(f"Best Case (95%):          ${result.final_equity_max:,.2f}")
        print(f"Worst Case (5%):          ${result.final_equity_min:,.2f}")
        
        print("\n--- DRAWDOWN RISK ---")
        print(f"Avg Max Drawdown:         {result.avg_max_drawdown:.2f}%")
        print(f"Worst Max Drawdown:       {result.worst_max_drawdown:.2f}%")
        print(f"Prob of DD > 20%:         {result.prob_dd_over_20*100:.1f}%")
        print(f"Prob of DD > 30%:         {result.prob_dd_over_30*100:.1f}%")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def compare_strategies(results: List[tuple]) -> pd.DataFrame:
        """
        Compare multiple Monte Carlo results
        
        Args:
            results: List of tuples (strategy_name, MonteCarloResult)
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for name, mc_result in results:
            comparison_data.append({
                'Strategy': name,
                'Expected Return %': mc_result.expected_return_percent,
                'Prob Profit %': mc_result.prob_profit * 100,
                'Risk of Ruin %': mc_result.prob_ruin * 100,
                'Avg Max DD %': mc_result.avg_max_drawdown,
                'Worst DD %': mc_result.worst_max_drawdown,
                'Final Equity (Mean)': mc_result.final_equity_mean,
                'Final Equity (Std)': mc_result.final_equity_std,
            })
        
        df = pd.DataFrame(comparison_data)
        return df.set_index('Strategy')

