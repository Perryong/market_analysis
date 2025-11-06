"""
Performance Metrics Calculator

Comprehensive performance and risk metrics for backtesting results including
Sharpe ratio, Sortino ratio, Calmar ratio, drawdowns, win rate, and more.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from backtesting.models import BacktestResult, Trade


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics container
    
    Contains all calculated performance and risk metrics from a backtest
    """
    # Return metrics
    total_return: float = 0.0
    total_return_percent: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    
    # Trading metrics
    num_trades: int = 0
    num_winning_trades: int = 0
    num_losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_percent: float = 0.0
    avg_loss_percent: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    avg_trade_duration_days: float = 0.0
    
    # Statistical metrics
    alpha: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    
    # Additional data
    daily_returns: pd.Series = field(default_factory=pd.Series)
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary (excluding series data)"""
        return {
            # Return metrics
            'total_return': self.total_return,
            'total_return_percent': self.total_return_percent,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            
            # Risk metrics
            'volatility': self.volatility,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            
            # Trading metrics
            'num_trades': self.num_trades,
            'num_winning_trades': self.num_winning_trades,
            'num_losing_trades': self.num_losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_win_percent': self.avg_win_percent,
            'avg_loss_percent': self.avg_loss_percent,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'expectancy': self.expectancy,
            'avg_trade_duration_days': self.avg_trade_duration_days,
            
            # Statistical metrics
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'value_at_risk_95': self.value_at_risk_95,
            'conditional_var_95': self.conditional_var_95,
        }


class MetricsCalculator:
    """Calculate comprehensive performance metrics from backtest results"""
    
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    
    @classmethod
    def calculate(cls, 
                  result: BacktestResult,
                  benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            result: BacktestResult from backtest run
            benchmark_returns: Optional benchmark returns for alpha/beta
            
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        metrics = PerformanceMetrics()
        
        # Skip if no data
        if len(result.equity_curve) == 0:
            return metrics
        
        # Calculate returns
        returns = result.get_returns()
        metrics.daily_returns = returns
        
        # Return metrics
        cls._calculate_return_metrics(result, returns, metrics)
        
        # Risk metrics
        cls._calculate_risk_metrics(result, returns, metrics)
        
        # Trading metrics
        cls._calculate_trading_metrics(result, metrics)
        
        # Statistical metrics
        if benchmark_returns is not None:
            cls._calculate_statistical_metrics(returns, benchmark_returns, metrics)
        
        # Monthly returns
        metrics.monthly_returns = cls._calculate_monthly_returns(result.equity_curve)
        
        return metrics
    
    @classmethod
    def _calculate_return_metrics(cls, result: BacktestResult,
                                  returns: pd.Series,
                                  metrics: PerformanceMetrics):
        """Calculate return-based metrics"""
        # Total return
        metrics.total_return = result.total_return
        metrics.total_return_percent = result.total_return_percent
        
        # Cumulative return
        metrics.cumulative_return = (result.final_equity / result.initial_capital) - 1
        
        # Annualized return
        years = result.duration_days / 365.0
        if years > 0:
            metrics.annualized_return = ((1 + metrics.cumulative_return) ** (1 / years)) - 1
        else:
            metrics.annualized_return = 0.0
    
    @classmethod
    def _calculate_risk_metrics(cls, result: BacktestResult,
                               returns: pd.Series,
                               metrics: PerformanceMetrics):
        """Calculate risk-based metrics"""
        if len(returns) == 0:
            return
        
        # Volatility
        metrics.volatility = float(returns.std())
        metrics.annualized_volatility = metrics.volatility * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
        
        # Sharpe Ratio
        if metrics.annualized_volatility > 0:
            excess_return = metrics.annualized_return - cls.RISK_FREE_RATE
            metrics.sharpe_ratio = excess_return / metrics.annualized_volatility
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = float(downside_returns.std())
            annualized_downside = downside_std * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
            if annualized_downside > 0:
                excess_return = metrics.annualized_return - cls.RISK_FREE_RATE
                metrics.sortino_ratio = excess_return / annualized_downside
        
        # Drawdown calculations
        cls._calculate_drawdowns(result.equity_curve, metrics)
        
        # Calmar Ratio
        if metrics.max_drawdown_percent != 0:
            metrics.calmar_ratio = abs(metrics.annualized_return / (metrics.max_drawdown_percent / 100))
        
        # Value at Risk (95% confidence)
        metrics.value_at_risk_95 = float(returns.quantile(0.05))
        
        # Conditional VaR (expected shortfall)
        worst_5_percent = returns[returns <= metrics.value_at_risk_95]
        if len(worst_5_percent) > 0:
            metrics.conditional_var_95 = float(worst_5_percent.mean())
    
    @classmethod
    def _calculate_drawdowns(cls, equity_curve: pd.Series, metrics: PerformanceMetrics):
        """Calculate drawdown metrics"""
        if len(equity_curve) == 0:
            return
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown in dollars
        drawdown = equity_curve - running_max
        metrics.drawdown_series = drawdown
        
        # Maximum drawdown
        metrics.max_drawdown = float(drawdown.min())
        metrics.max_drawdown_percent = (metrics.max_drawdown / float(running_max.max())) * 100
        
        # Average drawdown
        drawdowns_only = drawdown[drawdown < 0]
        if len(drawdowns_only) > 0:
            metrics.avg_drawdown = float(drawdowns_only.mean())
        
        # Maximum drawdown duration
        # Find periods where we're in drawdown
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # Find consecutive drawdown periods
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            if drawdown_periods:
                metrics.max_drawdown_duration_days = max(drawdown_periods)
    
    @classmethod
    def _calculate_trading_metrics(cls, result: BacktestResult, metrics: PerformanceMetrics):
        """Calculate trading-related metrics"""
        closed_trades = result.closed_trades
        
        metrics.num_trades = len(closed_trades)
        
        if len(closed_trades) == 0:
            return
        
        # Winning and losing trades
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]
        
        metrics.num_winning_trades = len(winning_trades)
        metrics.num_losing_trades = len(losing_trades)
        
        # Win rate
        metrics.win_rate = (metrics.num_winning_trades / metrics.num_trades) * 100
        
        # Average wins and losses
        if winning_trades:
            wins = [t.pnl for t in winning_trades if t.pnl]
            metrics.avg_win = np.mean(wins)
            metrics.largest_win = max(wins)
            
            win_percents = [t.pnl_percent for t in winning_trades if t.pnl_percent]
            if win_percents:
                metrics.avg_win_percent = np.mean(win_percents)
        
        if losing_trades:
            losses = [abs(t.pnl) for t in losing_trades if t.pnl]
            metrics.avg_loss = np.mean(losses)
            metrics.largest_loss = max(losses)
            
            loss_percents = [abs(t.pnl_percent) for t in losing_trades if t.pnl_percent]
            if loss_percents:
                metrics.avg_loss_percent = np.mean(loss_percents)
        
        # Profit factor
        if losing_trades and winning_trades:
            gross_profit = sum(t.pnl for t in winning_trades if t.pnl)
            gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl))
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss
        
        # Expectancy
        if metrics.num_trades > 0:
            total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
            metrics.expectancy = total_pnl / metrics.num_trades
        
        # Average trade duration
        durations = [t.duration_days for t in closed_trades if t.duration_days is not None]
        if durations:
            metrics.avg_trade_duration_days = np.mean(durations)
    
    @classmethod
    def _calculate_statistical_metrics(cls, returns: pd.Series,
                                       benchmark_returns: pd.Series,
                                       metrics: PerformanceMetrics):
        """Calculate statistical metrics vs benchmark"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return
        
        # Align dates
        aligned = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return
        
        strategy_returns = aligned['strategy']
        bench_returns = aligned['benchmark']
        
        # Calculate covariance and variance
        covariance = strategy_returns.cov(bench_returns)
        benchmark_variance = bench_returns.var()
        
        # Beta
        if benchmark_variance > 0:
            metrics.beta = covariance / benchmark_variance
        
        # Alpha (annualized)
        benchmark_annualized = (1 + bench_returns.mean()) ** cls.TRADING_DAYS_PER_YEAR - 1
        metrics.alpha = metrics.annualized_return - (cls.RISK_FREE_RATE + metrics.beta * (benchmark_annualized - cls.RISK_FREE_RATE))
        
        # R-squared
        correlation = strategy_returns.corr(bench_returns)
        metrics.r_squared = correlation ** 2
    
    @classmethod
    def _calculate_monthly_returns(cls, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns from equity curve"""
        if len(equity_curve) == 0:
            return pd.Series()
        
        # Resample to month end
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change()
        
        return monthly_returns
    
    @staticmethod
    def print_metrics(metrics: PerformanceMetrics):
        """Print formatted metrics summary"""
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        print("\n--- RETURN METRICS ---")
        print(f"Total Return:        ${metrics.total_return:,.2f} ({metrics.total_return_percent:.2f}%)")
        print(f"Annualized Return:   {metrics.annualized_return*100:.2f}%")
        print(f"Cumulative Return:   {metrics.cumulative_return*100:.2f}%")
        
        print("\n--- RISK METRICS ---")
        print(f"Volatility (Annual): {metrics.annualized_volatility*100:.2f}%")
        print(f"Sharpe Ratio:        {metrics.sharpe_ratio:.3f}")
        print(f"Sortino Ratio:       {metrics.sortino_ratio:.3f}")
        print(f"Calmar Ratio:        {metrics.calmar_ratio:.3f}")
        print(f"Max Drawdown:        ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        print(f"Avg Drawdown:        ${metrics.avg_drawdown:,.2f}")
        print(f"Max DD Duration:     {metrics.max_drawdown_duration_days} days")
        
        print("\n--- TRADING METRICS ---")
        print(f"Total Trades:        {metrics.num_trades}")
        print(f"Winning Trades:      {metrics.num_winning_trades}")
        print(f"Losing Trades:       {metrics.num_losing_trades}")
        print(f"Win Rate:            {metrics.win_rate:.1f}%")
        print(f"Profit Factor:       {metrics.profit_factor:.2f}")
        print(f"Average Win:         ${metrics.avg_win:,.2f} ({metrics.avg_win_percent:.2f}%)")
        print(f"Average Loss:        ${metrics.avg_loss:,.2f} ({metrics.avg_loss_percent:.2f}%)")
        print(f"Largest Win:         ${metrics.largest_win:,.2f}")
        print(f"Largest Loss:        ${metrics.largest_loss:,.2f}")
        print(f"Expectancy:          ${metrics.expectancy:.2f}")
        print(f"Avg Trade Duration:  {metrics.avg_trade_duration_days:.1f} days")
        
        print("\n--- STATISTICAL METRICS ---")
        print(f"Alpha:               {metrics.alpha*100:.2f}%")
        print(f"Beta:                {metrics.beta:.3f}")
        print(f"R-Squared:           {metrics.r_squared:.3f}")
        print(f"VaR (95%):           {metrics.value_at_risk_95*100:.2f}%")
        print(f"CVaR (95%):          {metrics.conditional_var_95*100:.2f}%")
        
        print("="*70 + "\n")

