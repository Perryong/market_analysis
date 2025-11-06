# Backtesting Framework Documentation

A comprehensive backtesting system for validating trading strategies using historical data, with advanced analytics including walk-forward analysis, Monte Carlo simulations, and parameter optimization.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Advanced Features](#advanced-features)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [API Reference](#api-reference)

## Overview

The backtesting framework provides:

- **Historical Performance Testing**: Simulate trading strategies on past data
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, win rate, profit factor, and more
- **Risk Assessment**: Monte Carlo simulations for probabilistic analysis
- **Strategy Validation**: Walk-forward analysis to detect overfitting
- **Parameter Optimization**: Grid search to find optimal strategy parameters
- **Professional Reports**: HTML, CSV, JSON exports with visualizations

## Installation

The backtesting framework is included in the main project. Ensure you have all dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- matplotlib
- yfinance (for data fetching)

## Quick Start

### Basic Backtest

```python
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.metrics import MetricsCalculator
from backtesting.visualizer import BacktestVisualizer
from backtesting.reports import BacktestReporter
from analysis.signal_generator import SignalAnalyzer
from config.settings import ScoringConfig

# Load configuration
config = ScoringConfig.from_yaml("config/weights.yaml")

# Create backtest configuration
backtest_config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,  # 0.1%
    slippage=0.0005,   # 0.05%
    position_size=0.1, # 10% per trade
    stop_loss_atr=2.0,
    take_profit_atr=3.0
)

# Initialize signal analyzer (your strategy)
signal_analyzer = SignalAnalyzer(config)
# ... register strategies ...

# Create backtest engine
engine = BacktestEngine(signal_analyzer, backtest_config)

# Run backtest
result = engine.run(
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2024-10-31"
)

# Calculate metrics
metrics = MetricsCalculator.calculate(result)

# Generate reports
reporter = BacktestReporter()
reporter.print_summary(result, metrics)
reporter.generate_html_report(result, metrics)

# Create visualizations
visualizer = BacktestVisualizer()
visualizer.create_full_report(result, metrics, "AAPL")
```

### Using the Integrated Function

```python
from crypto_analysis import backtest_strategy

# Simple backtest
backtest_strategy(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2024-10-31"
)
```

## Core Components

### 1. BacktestEngine

The core simulation engine that processes historical data and executes trades.

**Key Features:**
- Day-by-day simulation
- Realistic trade execution (commission, slippage)
- Position management (stop loss, take profit)
- Multiple position support

**Configuration Parameters:**
- `initial_capital`: Starting capital
- `commission`: Commission per trade (%)
- `slippage`: Slippage per trade (%)
- `position_size`: Position size as fraction of equity
- `max_positions`: Maximum concurrent positions
- `stop_loss_atr`: Stop loss in ATR multiples
- `take_profit_atr`: Take profit in ATR multiples
- `holding_period_days`: Maximum holding period

### 2. Performance Metrics

Comprehensive performance analysis with 30+ metrics.

**Return Metrics:**
- Total Return (%)
- Annualized Return
- Cumulative Return

**Risk Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Volatility (annualized)
- Value at Risk (VaR)
- Conditional VaR (CVaR)

**Trading Metrics:**
- Win Rate
- Profit Factor
- Average Win/Loss
- Expectancy
- Trade Duration

### 3. Visualization

Professional charts and visualizations:

- **Equity Curve**: Portfolio value over time with drawdown shading
- **Returns Distribution**: Histogram of daily returns
- **Monthly Returns Heatmap**: Performance by month and year
- **Trade Analysis**: P&L distribution, duration, win/loss breakdown
- **Monte Carlo Paths**: Probabilistic scenarios with confidence bands

### 4. Reports

Multiple export formats:

- **HTML Report**: Comprehensive web report with embedded metrics
- **CSV Export**: Trade-by-trade details
- **JSON Export**: Structured metrics data
- **Strategy Comparison**: Side-by-side comparison table

## Advanced Features

### Monte Carlo Simulation

Assess risk through resampling and simulation:

```python
from backtesting.monte_carlo import MonteCarloSimulator

# Run backtest first
result = engine.run("AAPL", "2023-01-01", "2024-10-31")

# Run Monte Carlo
simulator = MonteCarloSimulator(num_simulations=1000)
mc_result = simulator.run(result)

# Analyze results
print(f"Probability of Profit: {mc_result.prob_profit*100:.1f}%")
print(f"Risk of Ruin: {mc_result.prob_ruin*100:.1f}%")
print(f"Expected Return: {mc_result.expected_return_percent:.2f}%")

# Visualize
visualizer.plot_monte_carlo(
    mc_result.equity_curves[:100],
    percentiles={
        5: mc_result.percentile_5,
        50: mc_result.percentile_50,
        95: mc_result.percentile_95
    }
)
```

**Output:**
- Probability of profit/loss
- Risk of ruin (>50% capital loss)
- Expected final equity with std deviation
- Drawdown probabilities
- Confidence intervals (5th, 50th, 95th percentiles)

### Walk-Forward Analysis

Validate strategy robustness and detect overfitting:

```python
from backtesting.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(signal_analyzer, backtest_config)

wf_result = analyzer.run(
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2024-10-31",
    train_days=365,  # 1 year training
    test_days=90     # 3 months testing
)

# Analyze results
print(f"Avg Train Return: {wf_result.avg_train_return:.2f}%")
print(f"Avg Test Return: {wf_result.avg_test_return:.2f}%")
print(f"Consistency Score: {wf_result.consistency_score:.3f}")
print(f"Overfitting Detected: {wf_result.overfitting_detected}")

# Export and visualize
analyzer.export_windows_csv(wf_result, "walk_forward_results.csv")
analyzer.plot_walk_forward_results(wf_result)
```

**Features:**
- Rolling window optimization
- Out-of-sample testing
- Overfitting detection
- Consistency scoring
- Window-by-window analysis

### Parameter Optimization

Find optimal strategy parameters using grid search:

```python
from backtesting.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(base_config, backtest_config)

# Define parameter grid
param_grid = {
    'buy_threshold': [0.60, 0.65, 0.70, 0.75],
    'sell_threshold': [0.10, 0.15, 0.20, 0.25],
    'stop_loss_atr': [1.5, 2.0, 2.5],
}

# Run optimization
opt_result = optimizer.optimize(
    ticker="AAPL",
    start_date="2023-01-01",
    end_date="2024-10-31",
    param_grid=param_grid,
    metric='sharpe_ratio'  # or 'total_return_percent', 'sortino_ratio', etc.
)

# Get best parameters
print("Best Parameters:", opt_result.best_params)
print(f"Best Sharpe: {opt_result.best_result.sharpe_ratio:.3f}")

# Visualize optimization surface
optimizer.plot_optimization_surface(
    opt_result,
    param_x='buy_threshold',
    param_y='sell_threshold',
    save_path='optimization_surface.png'
)

# Export all results
optimizer.export_results_csv(opt_result, 'optimization_results.csv')
```

**Supported Metrics:**
- `sharpe_ratio`
- `sortino_ratio`
- `calmar_ratio`
- `total_return_percent`
- `profit_factor`
- `win_rate`

## Configuration

### YAML Configuration (config/weights.yaml)

```yaml
# Backtesting Configuration
backtesting:
  initial_capital: 100000     # Starting capital
  commission: 0.001           # 0.1% commission per trade
  slippage: 0.0005            # 0.05% slippage per trade
  position_size: 0.1          # 10% of equity per position
  max_positions: 10           # Maximum concurrent positions
  stop_loss_atr: 2.0          # Stop loss distance in ATR multiples
  take_profit_atr: 3.0        # Take profit distance in ATR multiples
  holding_period_days: null   # Max holding period (null = unlimited)

# Walk-Forward Analysis Configuration
walk_forward:
  train_days: 365             # Training window size
  test_days: 90               # Testing window size

# Monte Carlo Simulation Configuration
monte_carlo:
  num_simulations: 1000       # Number of simulation runs
  confidence_levels:          # Confidence levels for percentiles
    - 0.05
    - 0.25
    - 0.50
    - 0.75
    - 0.95
```

### Programmatic Configuration

```python
from backtesting.engine import BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005,
    position_size=0.1,
    max_positions=10,
    stop_loss_atr=2.0,
    take_profit_atr=3.0,
    holding_period_days=None
)
```

## Examples

### Example 1: Simple Backtest with Reports

```python
from crypto_analysis import backtest_strategy

backtest_strategy(
    tickers=["AAPL", "MSFT"],
    start_date="2022-01-01",
    end_date="2024-10-31",
    run_monte_carlo=False,
    run_walk_forward=False
)
```

**Output:**
- Console summary with all metrics
- HTML report in `results/backtesting/`
- Equity curve charts
- Trade-by-trade CSV
- Metrics JSON

### Example 2: Risk Assessment with Monte Carlo

```python
backtest_strategy(
    tickers=["NVDA"],
    start_date="2023-01-01",
    end_date="2024-10-31",
    run_monte_carlo=True,  # Enable Monte Carlo
    run_walk_forward=False
)
```

**Additional Output:**
- Probability distributions
- Risk metrics
- Monte Carlo confidence bands
- Drawdown probabilities

### Example 3: Strategy Validation

```python
backtest_strategy(
    tickers=["GOOGL"],
    start_date="2022-01-01",
    end_date="2024-10-31",
    run_monte_carlo=False,
    run_walk_forward=True  # Enable walk-forward
)
```

**Additional Output:**
- Window-by-window results
- Train vs test performance
- Overfitting detection
- Consistency metrics

### Example 4: Complete Analysis

```python
# Run all backtest examples
from crypto_analysis import run_backtest_example

run_backtest_example()
```

This runs:
1. Simple backtest on AAPL and MSFT
2. Monte Carlo simulation on NVDA
3. Walk-forward analysis on GOOGL
4. Parameter optimization on AAPL

## API Reference

### BacktestEngine

```python
class BacktestEngine:
    def __init__(self, signal_analyzer: SignalAnalyzer, 
                 config: BacktestConfig)
    
    def run(self, ticker: str, start_date: str, end_date: str,
            timeframe: TimeFrame = TimeFrame.DAILY) -> BacktestResult
```

### MetricsCalculator

```python
class MetricsCalculator:
    @classmethod
    def calculate(cls, result: BacktestResult,
                  benchmark_returns: Optional[pd.Series] = None
                 ) -> PerformanceMetrics
```

### BacktestVisualizer

```python
class BacktestVisualizer:
    def plot_equity_curve(self, result: BacktestResult, 
                         metrics: PerformanceMetrics)
    
    def plot_returns_distribution(self, metrics: PerformanceMetrics)
    
    def plot_monthly_returns_heatmap(self, metrics: PerformanceMetrics)
    
    def plot_trade_analysis(self, result: BacktestResult)
    
    def plot_monte_carlo(self, equity_curves: List[pd.Series],
                        percentiles: Dict)
    
    def create_full_report(self, result: BacktestResult,
                          metrics: PerformanceMetrics, ticker: str)
```

### BacktestReporter

```python
class BacktestReporter:
    def print_summary(self, result: BacktestResult, 
                     metrics: PerformanceMetrics)
    
    def export_trades_csv(self, result: BacktestResult, 
                         filename: Optional[str] = None)
    
    def export_metrics_json(self, metrics: PerformanceMetrics,
                           result: BacktestResult, 
                           filename: Optional[str] = None)
    
    def generate_html_report(self, result: BacktestResult,
                            metrics: PerformanceMetrics, 
                            filename: Optional[str] = None)
    
    @staticmethod
    def compare_strategies(results: List[tuple]) -> pd.DataFrame
```

### MonteCarloSimulator

```python
class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 1000,
                 confidence_levels: List[float] = None)
    
    def run(self, result: BacktestResult) -> MonteCarloResult
```

### WalkForwardAnalyzer

```python
class WalkForwardAnalyzer:
    def __init__(self, signal_analyzer: SignalAnalyzer,
                 config: BacktestConfig)
    
    def run(self, ticker: str, start_date: str, end_date: str,
            train_days: int = 365, test_days: int = 90,
            timeframe: TimeFrame = TimeFrame.DAILY
           ) -> WalkForwardResult
    
    def export_windows_csv(self, result: WalkForwardResult, 
                          filepath: str)
    
    def plot_walk_forward_results(self, result: WalkForwardResult)
```

### StrategyOptimizer

```python
class StrategyOptimizer:
    def __init__(self, base_config: ScoringConfig,
                 backtest_config: BacktestConfig)
    
    def optimize(self, ticker: str, start_date: str, end_date: str,
                param_grid: Dict[str, List[Any]], 
                metric: str = 'sharpe_ratio',
                timeframe: TimeFrame = TimeFrame.DAILY,
                max_combinations: Optional[int] = None
               ) -> GridSearchResult
    
    def plot_optimization_surface(self, result: GridSearchResult,
                                 param_x: str, param_y: str,
                                 save_path: Optional[str] = None)
    
    def export_results_csv(self, result: GridSearchResult, 
                          filepath: str)
```

## Output Structure

```
results/backtesting/
├── 2024-11-06/
│   ├── AAPL_backtest.html          # HTML report
│   ├── AAPL_trades.csv             # Trade history
│   ├── AAPL_metrics.json           # Metrics data
│   ├── AAPL_equity_curve.png       # Equity curve chart
│   ├── AAPL_returns_dist.png       # Returns distribution
│   ├── AAPL_monthly_returns.png    # Monthly heatmap
│   ├── AAPL_trade_analysis.png     # Trade analysis
│   ├── AAPL_monte_carlo.png        # Monte Carlo simulation
│   ├── AAPL_walk_forward.png       # Walk-forward analysis
│   ├── AAPL_walk_forward.csv       # Walk-forward windows
│   ├── AAPL_optimization.csv       # Optimization results
│   └── AAPL_optimization_surface.png
```

## Best Practices

1. **Use Sufficient Historical Data**: At least 2+ years for reliable results
2. **Realistic Costs**: Include commission and slippage
3. **Validate with Walk-Forward**: Always check for overfitting
4. **Risk Assessment**: Run Monte Carlo for probability analysis
5. **Parameter Robustness**: Test parameter sensitivity
6. **Compare to Benchmark**: Use buy-and-hold as baseline
7. **Document Everything**: Save configurations and results

## Troubleshooting

### Common Issues

**Issue: "Insufficient data for ticker"**
- Solution: Ensure ticker symbol is correct and data is available for the date range

**Issue: "No closed trades to analyze"**
- Solution: Strategy may not be generating signals. Check signal thresholds and parameters

**Issue: "Optimization taking too long"**
- Solution: Reduce parameter grid size or use `max_combinations` parameter

**Issue: "Walk-forward windows failing"**
- Solution: Ensure date range is sufficient for multiple windows (train + test periods)

## Support

For issues, questions, or contributions:
- Check examples in `crypto_analysis.py`
- Review configuration in `config/weights.yaml`
- Examine module docstrings for detailed usage

## License

Part of the Trading Strategy Analysis System - All Rights Reserved

