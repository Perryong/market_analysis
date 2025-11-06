"""
Backtesting Visualization

Creates comprehensive charts for backtest results including equity curves,
drawdowns, returns distributions, and Monte Carlo simulations.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

from backtesting.models import BacktestResult
from backtesting.metrics import PerformanceMetrics


class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    def __init__(self, output_dir: str = "results/backtesting"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory for saving charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_equity_curve(self,
                         result: BacktestResult,
                         metrics: PerformanceMetrics,
                         save_path: Optional[str] = None,
                         show: bool = False):
        """
        Plot equity curve with drawdown shading
        
        Args:
            result: Backtest result
            metrics: Performance metrics
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if len(result.equity_curve) == 0:
            print("[WARNING] No equity data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        dates = result.equity_curve.index
        equity = result.equity_curve.values
        
        ax1.plot(dates, equity, linewidth=2, color='#2E86AB', label='Portfolio Equity')
        ax1.axhline(y=result.initial_capital, color='gray', linestyle='--', 
                   linewidth=1, label='Initial Capital', alpha=0.7)
        
        # Shade drawdown periods
        running_max = pd.Series(equity).expanding().max().values
        drawdown_pct = ((equity - running_max) / running_max) * 100
        
        # Fill area for drawdowns
        ax1.fill_between(dates, equity, running_max, 
                        where=(equity < running_max),
                        color='red', alpha=0.2, label='Drawdown')
        
        # Format ax1
        ax1.set_title(f'Equity Curve - {result.ticker}\n'
                     f'Return: {metrics.total_return_percent:.2f}% | '
                     f'Sharpe: {metrics.sharpe_ratio:.2f} | '
                     f'Max DD: {metrics.max_drawdown_percent:.2f}%',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot drawdown percentage
        ax2.fill_between(dates, drawdown_pct, 0, 
                        where=(drawdown_pct <= 0),
                        color='red', alpha=0.5, label='Drawdown %')
        ax2.plot(dates, drawdown_pct, linewidth=1, color='darkred')
        
        # Format ax2
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_ylim([min(drawdown_pct.min() * 1.1, -1), 1])
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[*] Saved equity curve to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_returns_distribution(self,
                                  metrics: PerformanceMetrics,
                                  save_path: Optional[str] = None,
                                  show: bool = False):
        """
        Plot returns distribution histogram
        
        Args:
            metrics: Performance metrics with returns data
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if len(metrics.daily_returns) == 0:
            print("[WARNING] No returns data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        returns = metrics.daily_returns * 100  # Convert to percentage
        
        # Plot histogram
        n, bins, patches = ax.hist(returns, bins=50, edgecolor='black', 
                                   alpha=0.7, color='#2E86AB')
        
        # Color bars based on positive/negative
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#E63946')  # Red for negative
            else:
                patch.set_facecolor('#06A77D')  # Green for positive
        
        # Add vertical lines for mean and median
        ax.axvline(returns.mean(), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean: {returns.mean():.3f}%')
        ax.axvline(returns.median(), color='orange', linestyle='--', 
                  linewidth=2, label=f'Median: {returns.median():.3f}%')
        
        # Add VaR line
        var_95 = returns.quantile(0.05)
        ax.axvline(var_95, color='red', linestyle=':', 
                  linewidth=2, label=f'VaR (95%): {var_95:.3f}%')
        
        # Format
        ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[*] Saved returns distribution to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_monthly_returns_heatmap(self,
                                     metrics: PerformanceMetrics,
                                     save_path: Optional[str] = None,
                                     show: bool = False):
        """
        Plot monthly returns as heatmap
        
        Args:
            metrics: Performance metrics with monthly returns
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if len(metrics.monthly_returns) == 0:
            print("[WARNING] No monthly returns data to plot")
            return
        
        # Prepare data
        monthly = metrics.monthly_returns * 100  # Convert to percentage
        
        # Create pivot table
        pivot_data = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })
        
        pivot = pivot_data.pivot(index='Month', columns='Year', values='Return')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                      vmin=-10, vmax=10)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color="black",
                                 fontsize=8, fontweight='bold')
        
        # Format
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Month', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[*] Saved monthly returns heatmap to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trade_analysis(self,
                           result: BacktestResult,
                           save_path: Optional[str] = None,
                           show: bool = False):
        """
        Plot trade analysis charts (P&L distribution, duration, etc.)
        
        Args:
            result: Backtest result
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        closed_trades = result.closed_trades
        
        if len(closed_trades) == 0:
            print("[WARNING] No closed trades to analyze")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. P&L Distribution
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        ax1.hist([wins, losses], bins=20, label=['Wins', 'Losses'],
                color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                linewidth=2, color='#2E86AB')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                        alpha=0.3, color='#2E86AB')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 3. Trade Duration Distribution
        durations = [t.duration_days for t in closed_trades if t.duration_days is not None]
        if durations:
            ax3.hist(durations, bins=20, color='#06A77D', 
                    alpha=0.7, edgecolor='black')
            ax3.set_title('Trade Duration Distribution', fontweight='bold')
            ax3.set_xlabel('Duration (Days)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss Statistics
        num_wins = len([p for p in pnls if p > 0])
        num_losses = len([p for p in pnls if p < 0])
        
        categories = ['Winners', 'Losers']
        values = [num_wins, num_losses]
        colors = ['green', 'red']
        
        ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Win/Loss Summary', fontweight='bold')
        ax4.set_ylabel('Number of Trades')
        
        # Add percentage labels
        total = num_wins + num_losses
        for i, (cat, val) in enumerate(zip(categories, values)):
            pct = (val / total) * 100 if total > 0 else 0
            ax4.text(i, val, f'{val}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[*] Saved trade analysis to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_monte_carlo(self,
                        equity_curves: List[pd.Series],
                        percentiles: Optional[dict] = None,
                        save_path: Optional[str] = None,
                        show: bool = False):
        """
        Plot Monte Carlo simulation results
        
        Args:
            equity_curves: List of simulated equity curves
            percentiles: Dict with percentile curves (e.g., {5: series, 50: series, 95: series})
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if len(equity_curves) == 0:
            print("[WARNING] No Monte Carlo data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot individual simulations (sample for clarity)
        sample_size = min(100, len(equity_curves))
        for curve in equity_curves[:sample_size]:
            ax.plot(curve.values, color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot percentiles if provided
        if percentiles:
            colors = {5: 'red', 50: 'blue', 95: 'green'}
            labels = {5: '5th Percentile (Worst)', 
                     50: '50th Percentile (Median)',
                     95: '95th Percentile (Best)'}
            
            for pct, curve in percentiles.items():
                ax.plot(curve.values, color=colors.get(pct, 'black'),
                       linewidth=2, label=labels.get(pct, f'{pct}th Percentile'))
        
        # Format
        ax.set_title(f'Monte Carlo Simulation ({len(equity_curves)} scenarios)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[*] Saved Monte Carlo plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_full_report(self,
                          result: BacktestResult,
                          metrics: PerformanceMetrics,
                          ticker: str):
        """
        Create all charts for a backtest result
        
        Args:
            result: Backtest result
            metrics: Performance metrics
            ticker: Ticker symbol for filenames
        """
        # Create dated directory
        today = datetime.now().strftime('%Y-%m-%d')
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] Generating backtest visualizations...")
        
        # Generate all charts
        self.plot_equity_curve(
            result, metrics,
            save_path=output_dir / f"{ticker}_equity_curve.png"
        )
        
        self.plot_returns_distribution(
            metrics,
            save_path=output_dir / f"{ticker}_returns_dist.png"
        )
        
        self.plot_monthly_returns_heatmap(
            metrics,
            save_path=output_dir / f"{ticker}_monthly_returns.png"
        )
        
        self.plot_trade_analysis(
            result,
            save_path=output_dir / f"{ticker}_trade_analysis.png"
        )
        
        print(f"[SUCCESS] All charts saved to {output_dir}/")

