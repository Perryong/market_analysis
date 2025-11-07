"""
Generate summary reports for TraderXO crypto charts
Similar to the stock market report generator
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class TraderXOReportGenerator:
    """Generate summary HTML reports for TraderXO crypto charts"""
    
    def __init__(self, plots_dir: str = "traderXO/plots"):
        """
        Initialize report generator
        
        Args:
            plots_dir: Base directory for traderXO plots
        """
        self.plots_dir = Path(plots_dir)
        self.date_str = datetime.now().strftime("%Y-%m-%d")
    
    def generate_summary_report(self, date_str: Optional[str] = None) -> Path:
        """
        Generate a summary HTML report for a specific date's crypto charts
        
        Args:
            date_str: Date string in YYYY-MM-DD format. If None, uses current date.
            
        Returns:
            Path to the generated report file
        """
        if date_str is None:
            date_str = self.date_str
        
        date_dir = self.plots_dir / date_str
        
        if not date_dir.exists():
            print(f"[WARNING] No plots directory found for {date_str}")
            return None
        
        # Find all PNG files in the date directory
        plot_files = sorted(date_dir.glob("*.png"))
        
        if not plot_files:
            print(f"[WARNING] No plots found for {date_str}")
            return None
        
        # Organize plots by ticker and strategy
        plots_by_ticker = {}
        for plot_file in plot_files:
            filename = plot_file.stem  # e.g., "BTC_USDT_1d_momentum"
            parts = filename.split('_')
            
            if len(parts) >= 3:
                ticker = f"{parts[0]}/{parts[1]}"  # e.g., "BTC/USDT"
                timeframe = parts[2]  # e.g., "1d"
                strategy = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                
                if ticker not in plots_by_ticker:
                    plots_by_ticker[ticker] = {}
                
                if strategy not in plots_by_ticker[ticker]:
                    plots_by_ticker[ticker][strategy] = []
                
                plots_by_ticker[ticker][strategy].append({
                    'file': plot_file.name,
                    'timeframe': timeframe,
                    'strategy': strategy
                })
        
        # Generate HTML report
        report_path = date_dir / "crypto_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Market Analysis - {date_str}</title>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .ticker-section {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .ticker-header {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .strategy-group {{
            margin: 15px 0;
        }}
        .strategy-title {{
            font-size: 1.1em;
            font-weight: bold;
            color: #666;
            margin-bottom: 10px;
            text-transform: capitalize;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}
        .plot-card {{
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
            text-decoration: none;
            color: inherit;
            display: block;
        }}
        .plot-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .plot-info {{
            font-size: 0.9em;
            color: #666;
        }}
        .timeframe-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <h1>🪙 Crypto Market Analysis Report</h1>
    <p>Generated: {date_str}</p>
    <p>Total Tickers: {len(plots_by_ticker)}</p>
"""
        
        # Generate HTML for each ticker
        for ticker in sorted(plots_by_ticker.keys()):
            ticker_data = plots_by_ticker[ticker]
            html_content += f"""
    <div class="ticker-section">
        <div class="ticker-header">{ticker}</div>
"""
            
            # Group by strategy
            for strategy in sorted(ticker_data.keys()):
                plots = ticker_data[strategy]
                strategy_display = strategy.replace('_', ' ').title()
                
                html_content += f"""
        <div class="strategy-group">
            <div class="strategy-title">{strategy_display}</div>
            <div class="plot-grid">
"""
                
                for plot in plots:
                    plot_name = plot['file']
                    timeframe = plot['timeframe']
                    html_content += f"""
                <a href="{plot_name}" class="plot-card" target="_blank">
                    <img src="{plot_name}" alt="{ticker} {strategy_display}" loading="lazy">
                    <div class="plot-info">
                        <span class="timeframe-badge">{timeframe}</span>
                        {strategy_display}
                    </div>
                </a>
"""
                
                html_content += """
            </div>
        </div>
"""
            
            html_content += """
    </div>
"""
        
        html_content += """
    </body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Crypto report saved: {report_path}")
        return report_path
    
    def generate_all_reports(self) -> List[Path]:
        """
        Generate reports for all available date directories
        
        Returns:
            List of paths to generated reports
        """
        if not self.plots_dir.exists():
            print(f"[WARNING] Plots directory not found: {self.plots_dir}")
            return []
        
        report_paths = []
        
        # Find all date directories
        for date_dir in sorted(self.plots_dir.iterdir(), reverse=True):
            if date_dir.is_dir() and date_dir.name.startswith("20"):  # Date folders start with year
                try:
                    report_path = self.generate_summary_report(date_dir.name)
                    if report_path:
                        report_paths.append(report_path)
                except Exception as e:
                    print(f"[ERROR] Failed to generate report for {date_dir.name}: {e}")
        
        return report_paths

