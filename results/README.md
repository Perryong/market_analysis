# Results Folder

This folder contains the trading signal analysis results exported as **CSV** and **JSON** files.

## File Naming Convention

All results are automatically saved with the current date prefix:

### CSV Files
- `YYYY-MM-DD_us_signals.csv` - US market analysis results (CSV format)
- `YYYY-MM-DD_sg_signals.csv` - Singapore market analysis results (CSV format)
- `YYYY-MM-DD_combined_signals.csv` - Combined results from both markets (CSV format)

### JSON Files
- `YYYY-MM-DD_us_signals.json` - US market analysis results (JSON format)
- `YYYY-MM-DD_sg_signals.json` - Singapore market analysis results (JSON format)
- `YYYY-MM-DD_combined_signals.json` - Combined results from both markets (JSON format)

## Example

When you run `python main.py` on November 5, 2025, the following files will be generated:

**CSV Files:**
- `2025-11-05_us_signals.csv`
- `2025-11-05_sg_signals.csv`
- `2025-11-05_combined_signals.csv`

**JSON Files:**
- `2025-11-05_us_signals.json`
- `2025-11-05_sg_signals.json`
- `2025-11-05_combined_signals.json`

## File Structures

### CSV Files

#### Individual Market Files (US/SG)

| Column | Description |
|--------|-------------|
| Ticker | Stock ticker symbol |
| Name | Company short name |
| Signal | BUY, SELL, or HOLD |
| Confidence % | Signal confidence percentage |
| Last Close | Most recent closing price |
| ATR | Average True Range (volatility measure) |
| Entry Low | Lower bound of entry zone |
| Entry High | Upper bound of entry zone |
| Timeframe | Analysis timeframe (DAILY) |

#### Combined CSV File

Same structure as above, but includes an additional `Market` column (US or SG) to distinguish the source.

### JSON Files

#### Individual Market JSON Structure

```json
{
  "report_date": "2025-11-05 01:01:00",
  "total_analyzed": 11,
  "summary": {
    "buy": 4,
    "sell": 0,
    "hold": 7
  },
  "signals": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "signal": "HOLD",
      "confidence": 36.2,
      "last_close": 269.05,
      "atr": 5.1834,
      "entry_zone": {
        "lower": 266.46,
        "upper": 273.20
      },
      "timeframe": "1d",
      "is_actionable": false,
      "analysis": [
        "RSI 66.5 - Bearish (trend: v)",
        "MACD Mild Bullish (spread: 0.7293, d_hist: -0.2638)",
        "OBV slope -22158390.7 - Strong Bearish volume flow",
        "MA50 249.07 vs MA200 223.51 - Strong Bullish Trend (spread: 11.4%)",
        "BB Mild Bearish (upper half) - price middle<->upper (wide bands, width: 0.136)",
        "Volume 50,194,583 vs SMA20 48,108,652 - Mild Bearish (normal volume) (ratio: 1.04x)",
        "Neutral Candlesticks 0.35 : bear_harami",
        "Weak trend (ADX=18.5) - reduced confidence",
        "Near S/R level at $260.10"
      ]
    }
  ]
}
```

#### Combined JSON Structure

```json
{
  "report_date": "2025-11-05 01:01:00",
  "total_analyzed": 23,
  "markets": {
    "us": {
      "total": 11,
      "buy": 4,
      "sell": 0,
      "hold": 7
    },
    "sg": {
      "total": 12,
      "buy": 1,
      "sell": 0,
      "hold": 11
    }
  },
  "summary": {
    "total_buy": 5,
    "total_sell": 0,
    "total_hold": 18
  },
  "us_signals": [...],
  "sg_signals": [...]
}
```

## Usage

Results are automatically generated when you run the main analysis:

```bash
python main.py
```

The system will create dated CSV files in this folder for easy tracking of historical analysis results.

## Note

CSV files are ignored in git (see `.gitignore`), but the folder structure is preserved.

