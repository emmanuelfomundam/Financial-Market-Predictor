# Stock Volatility Predictor

## Situation
Financial markets are characterized by volatility, which represents a significant risk factor for investors. Traditional volatility models like GARCH and modern deep learning approaches (LSTM) offer different methodologies for forecasting market volatility. This project addresses the need for an accessible tool that allows users to compare these approaches using real market data.

## Task
Develop a desktop application that:
1. Downloads historical stock data
2. Calculates realized volatility
3. Forecasts volatility using both GARCH(1,1) and LSTM models
4. Compares model performance through visualizations and metrics
5. Provides an intuitive GUI for non-technical users

## Action
### Solution Overview
Built a Python application that:
- Uses Yahoo Finance API for data acquisition
- Implements GARCH(1,1) for traditional volatility modeling
- Creates LSTM neural network for deep learning approach
- Features Tkinter GUI with:
  - Company/ticker selection (500+ preloaded options)
  - Interactive parameter controls
  - Performance metrics comparison
  - Visualization dashboard
  - Logging system
  - Export capabilities

### Key Features:
1. **Data Processing:**
   - Real-time data download from Yahoo Finance
   - Rolling volatility calculation with adjustable window
   - Returns normalization for model input

2. **Modeling:**
   - **GARCH(1,1):** Implements ARCH library with walk-forward forecasting
   - **LSTM Network:** 2-layer architecture with dropout regularization
   - Min-Max scaling for volatility sequences

3. **Evaluation:**
   - Comparative metrics (MSE, RMSE, MAE, MAPE)
   - Side-by-side model performance visualization
   - Interactive volatility vs. price chart

4. **User Experience:**
   - Progress tracking with cancel functionality
   - Threaded execution to maintain responsive UI
   - Export results to CSV
   - Save visualizations as PNG/PDF
   - Detailed logging system

## Result
The application successfully:
- Provides comparative analysis of GARCH vs. LSTM volatility forecasting
- Delivers intuitive visualizations of market data and predictions
- Achieves responsive performance through threaded operations
- Offers practical value for investors and financial analysts
- Handles errors gracefully with user-friendly messaging

### Sample Output Metrics:
| Metric          | GARCH(1,1) | LSTM    |
|-----------------|------------|---------|
| MSE             | 0.000215   | 0.000193|
| RMSE            | 0.014657   | 0.013882|
| MAE             | 0.011028   | 0.010576|
| MAPE            | 12.45%     | 11.92%  |

### Prerequisites
- Python 3.7+
- Libraries in `requirements.txt`

### Installation
```bash
git clone https://github.com/yourusername/Stock-Volatility-Predictor.git
cd Stock-Volatility-Predictor
pip install -r requirements.txt
