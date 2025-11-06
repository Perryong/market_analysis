"""
Backtesting Reports and Exports

Generate comprehensive reports including console summaries, HTML reports,
CSV/JSON exports, and strategy comparison tools.
"""

from typing import List, Optional, Dict
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from backtesting.models import BacktestResult, Trade
from backtesting.metrics import PerformanceMetrics


class BacktestReporter:
    """Generate reports and exports for backtest results"""
    
    def __init__(self, output_dir: str = "results/backtesting"):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def print_summary(self, result: BacktestResult, metrics: PerformanceMetrics):
        """
        Print comprehensive summary to console
        
        Args:
            result: Backtest result
            metrics: Performance metrics
        """
        print(f"\n{'='*70}")
        print(f"BACKTEST REPORT: {result.ticker}")
        print(f"{'='*70}")
        
        # Period and setup
        print(f"\n--- BACKTEST SETUP ---")
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Duration: {result.duration_days} days")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Commission: {result.commission*100:.3f}%")
        print(f"Slippage: {result.slippage*100:.3f}%")
        print(f"Position Size: {result.position_size*100:.1f}%")
        
        # Returns
        print(f"\n--- RETURNS ---")
        print(f"Final Equity: ${result.final_equity:,.2f}")
        print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)")
        print(f"Annualized Return: {metrics.annualized_return*100:.2f}%")
        print(f"Cumulative Return: {metrics.cumulative_return*100:.2f}%")
        
        # Risk
        print(f"\n--- RISK METRICS ---")
        print(f"Volatility (Annual): {metrics.annualized_volatility*100:.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        print(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        print(f"Avg Drawdown: ${metrics.avg_drawdown:,.2f}")
        print(f"Max DD Duration: {metrics.max_drawdown_duration_days} days")
        print(f"VaR (95%): {metrics.value_at_risk_95*100:.2f}%")
        print(f"CVaR (95%): {metrics.conditional_var_95*100:.2f}%")
        
        # Trading
        print(f"\n--- TRADING STATISTICS ---")
        print(f"Total Trades: {metrics.num_trades}")
        print(f"Winning Trades: {metrics.num_winning_trades} ({metrics.win_rate:.1f}%)")
        print(f"Losing Trades: {metrics.num_losing_trades}")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Expectancy: ${metrics.expectancy:.2f}")
        print(f"Avg Win: ${metrics.avg_win:,.2f} ({metrics.avg_win_percent:.2f}%)")
        print(f"Avg Loss: ${metrics.avg_loss:,.2f} ({metrics.avg_loss_percent:.2f}%)")
        print(f"Largest Win: ${metrics.largest_win:,.2f}")
        print(f"Largest Loss: ${metrics.largest_loss:,.2f}")
        print(f"Avg Trade Duration: {metrics.avg_trade_duration_days:.1f} days")
        
        print(f"\n{'='*70}\n")
    
    def export_trades_csv(self, result: BacktestResult, filename: Optional[str] = None):
        """
        Export trade history to CSV
        
        Args:
            result: Backtest result
            filename: Optional custom filename
        """
        if not filename:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"{today}_{result.ticker}_trades.csv"
        
        filepath = self.output_dir / filename
        
        # Convert trades to DataFrame
        trades_data = [t.to_dict() for t in result.trades]
        df = pd.DataFrame(trades_data)
        
        # Sort by entry date
        if 'entry_date' in df.columns:
            df = df.sort_values('entry_date')
        
        df.to_csv(filepath, index=False)
        print(f"[*] Exported trades to {filepath}")
        
        return filepath
    
    def export_metrics_json(self, metrics: PerformanceMetrics, 
                           result: BacktestResult,
                           filename: Optional[str] = None):
        """
        Export metrics to JSON
        
        Args:
            metrics: Performance metrics
            result: Backtest result
            filename: Optional custom filename
        """
        if not filename:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"{today}_{result.ticker}_metrics.json"
        
        filepath = self.output_dir / filename
        
        # Combine metrics and summary
        data = {
            'backtest_info': result.to_summary_dict(),
            'performance_metrics': metrics.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"[*] Exported metrics to {filepath}")
        
        return filepath
    
    def generate_html_report(self, 
                            result: BacktestResult,
                            metrics: PerformanceMetrics,
                            include_charts: bool = True,
                            filename: Optional[str] = None):
        """
        Generate comprehensive HTML report
        
        Args:
            result: Backtest result
            metrics: Performance metrics
            include_charts: Whether to embed chart images
            filename: Optional custom filename
        """
        if not filename:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"{today}_{result.ticker}_report.html"
        
        filepath = self.output_dir / filename
        
        # Create HTML content
        html = self._generate_html_content(result, metrics, include_charts)
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        print(f"[*] Generated HTML report: {filepath}")
        
        return filepath
    
    def _generate_html_content(self, 
                               result: BacktestResult,
                               metrics: PerformanceMetrics,
                               include_charts: bool) -> str:
        """Generate HTML report content"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {result.ticker}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Backtest Report</h1>
        <p><strong>Ticker:</strong> {result.ticker}</p>
        <p><strong>Period:</strong> {result.start_date.date()} to {result.end_date.date()} ({result.duration_days} days)</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📈 Performance Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if result.total_return > 0 else 'negative'}">
                    {result.total_return_percent:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Annualized Return</div>
                <div class="metric-value {'positive' if metrics.annualized_return > 0 else 'negative'}">
                    {metrics.annualized_return*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics.sharpe_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics.max_drawdown_percent:.2f}%</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>💰 Return Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Initial Capital</td>
                <td>${result.initial_capital:,.2f}</td>
            </tr>
            <tr>
                <td>Final Equity</td>
                <td>${result.final_equity:,.2f}</td>
            </tr>
            <tr>
                <td>Total Return</td>
                <td>${result.total_return:,.2f} ({result.total_return_percent:.2f}%)</td>
            </tr>
            <tr>
                <td>Annualized Return</td>
                <td>{metrics.annualized_return*100:.2f}%</td>
            </tr>
            <tr>
                <td>Cumulative Return</td>
                <td>{metrics.cumulative_return*100:.2f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Volatility (Annualized)</td>
                <td>{metrics.annualized_volatility*100:.2f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{metrics.sharpe_ratio:.3f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{metrics.sortino_ratio:.3f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{metrics.calmar_ratio:.3f}</td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td>{metrics.max_drawdown_percent:.2f}%</td>
            </tr>
            <tr>
                <td>Average Drawdown</td>
                <td>${metrics.avg_drawdown:,.2f}</td>
            </tr>
            <tr>
                <td>Max DD Duration</td>
                <td>{metrics.max_drawdown_duration_days} days</td>
            </tr>
            <tr>
                <td>Value at Risk (95%)</td>
                <td>{metrics.value_at_risk_95*100:.2f}%</td>
            </tr>
            <tr>
                <td>Conditional VaR (95%)</td>
                <td>{metrics.conditional_var_95*100:.2f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>🎯 Trading Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{metrics.num_trades}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{metrics.num_winning_trades}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{metrics.num_losing_trades}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{metrics.win_rate:.1f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{metrics.profit_factor:.2f}</td>
            </tr>
            <tr>
                <td>Expectancy</td>
                <td>${metrics.expectancy:.2f}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>${metrics.avg_win:,.2f} ({metrics.avg_win_percent:.2f}%)</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>${metrics.avg_loss:,.2f} ({metrics.avg_loss_percent:.2f}%)</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td>${metrics.largest_win:,.2f}</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td>${metrics.largest_loss:,.2f}</td>
            </tr>
            <tr>
                <td>Average Trade Duration</td>
                <td>{metrics.avg_trade_duration_days:.1f} days</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>⚙️ Backtest Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Commission</td>
                <td>{result.commission*100:.3f}%</td>
            </tr>
            <tr>
                <td>Slippage</td>
                <td>{result.slippage*100:.3f}%</td>
            </tr>
            <tr>
                <td>Position Size</td>
                <td>{result.position_size*100:.1f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by Trading Strategy Backtesting Engine</p>
        <p>&copy; {datetime.now().year} - All Rights Reserved</p>
    </div>
</body>
</html>
"""
        
        return html
    
    @staticmethod
    def compare_strategies(results: List[tuple]) -> pd.DataFrame:
        """
        Compare multiple backtest results
        
        Args:
            results: List of tuples (strategy_name, BacktestResult, PerformanceMetrics)
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for name, result, metrics in results:
            comparison_data.append({
                'Strategy': name,
                'Total Return %': result.total_return_percent,
                'Annualized Return %': metrics.annualized_return * 100,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Calmar Ratio': metrics.calmar_ratio,
                'Max Drawdown %': metrics.max_drawdown_percent,
                'Win Rate %': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
                'Num Trades': metrics.num_trades,
                'Avg Trade Duration': metrics.avg_trade_duration_days,
            })
        
        df = pd.DataFrame(comparison_data)
        return df.set_index('Strategy')
    
    def export_comparison_csv(self, comparison_df: pd.DataFrame, filename: str = "strategy_comparison.csv"):
        """Export strategy comparison to CSV"""
        filepath = self.output_dir / filename
        comparison_df.to_csv(filepath)
        print(f"[*] Exported comparison to {filepath}")
        
        return filepath

