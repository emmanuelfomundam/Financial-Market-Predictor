import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import datetime
import warnings
import matplotlib
import threading
import re
matplotlib.use('TkAgg')

# Suppress warnings
warnings.filterwarnings('ignore')

# Extended popular tickers for dropdown
POPULAR_TICKERS = {
    # Tech
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "Meta Platforms (Facebook)": "META",
    "NVIDIA": "NVDA",
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "Intel": "INTC",
    "AMD": "AMD",
    "Cisco": "CSCO",
    "Salesforce": "CRM",
    "IBM": "IBM",
    "Oracle": "ORCL",
    "Palantir": "PLTR",
    "Snowflake": "SNOW",
    
    # Automotive & EV
    "Tesla": "TSLA",
    "Ford": "F",
    "General Motors": "GM",
    "Toyota": "TM",
    "Nio": "NIO",
    "Lucid Motors": "LCID",
    "Rivian": "RIVN",

    # Financial
    "JPMorgan Chase": "JPM",
    "Bank of America": "BAC",
    "Wells Fargo": "WFC",
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "Citigroup": "C",
    "American Express": "AXP",
    "Charles Schwab": "SCHW",

    # Retail
    "Walmart": "WMT",
    "Target": "TGT",
    "Costco": "COST",
    "Home Depot": "HD",
    "Lowe's": "LOW",
    "Best Buy": "BBY",
    "Macy's": "M",

    # Energy
    "ExxonMobil": "XOM",
    "Chevron": "CVX",
    "Occidental Petroleum": "OXY",
    "ConocoPhillips": "COP",
    "NextEra Energy": "NEE",

    # Healthcare
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "Merck": "MRK",
    "UnitedHealth Group": "UNH",
    "AbbVie": "ABBV",
    "Bristol-Myers Squibb": "BMY",
    "CVS Health": "CVS",

    # Consumer Goods
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Procter & Gamble": "PG",
    "Unilever": "UL",
    "Colgate-Palmolive": "CL",
    "Mondelez": "MDLZ",

    # Airlines
    "Delta Air Lines": "DAL",
    "American Airlines": "AAL",
    "United Airlines": "UAL",
    "Southwest Airlines": "LUV",

    # Real Estate & REITs
    "Realty Income": "O",
    "Simon Property Group": "SPG",
    "Equinix": "EQIX",

    # Entertainment/Media
    "Disney": "DIS",
    "Warner Bros. Discovery": "WBD",
    "Comcast": "CMCSA",
    "Spotify": "SPOT",

    # Telecom
    "AT&T": "T",
    "Verizon": "VZ",
    "T-Mobile": "TMUS",

    # Industrials
    "3M": "MMM",
    "Caterpillar": "CAT",
    "General Electric": "GE",
    "Boeing": "BA",
    "Deere & Co": "DE",
    "Lockheed Martin": "LMT",
    "Northrop Grumman": "NOC",

    # Logistics
    "FedEx": "FDX",
    "UPS": "UPS",

    # Crypto-related
    "Coinbase": "COIN",
    "Marathon Digital": "MARA",
    "Riot Platforms": "RIOT"
}

class StopTrainingCallback(Callback):
    """Custom callback to stop training if requested"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
    
    def on_epoch_end(self, epoch, logs=None):
        if self.parent.stop_analysis:
            self.model.stop_training = True

class VolatilityPredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Volatility Predictor")
        self.root.geometry("1400x900")  # Wider window for side-by-side layout
        self.root.configure(bg="#f0f2f5")
        
        # Setup styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background="#f0f2f5")
        self.style.configure('TLabel', background="#f0f2f5", font=('Arial', 10))
        self.style.configure('Header.TLabel', background="#1a3d66", foreground="white", 
                            font=('Arial', 16, 'bold'), padding=10)
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Run.TButton', background="#27ae60", foreground="white", 
                           font=('Arial', 12, 'bold'), padding=5)
        self.style.configure('TCombobox', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))
        self.style.configure('TNotebook', background="#f0f2f5")
        self.style.configure('TNotebook.Tab', background="#d1d9e0", padding=(10, 5))
        self.style.map('TNotebook.Tab', background=[('selected', '#1a3d66')], 
                      foreground=[('selected', 'white')])
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        plt.close('all')
        self.root.destroy()
    
    def validate_numeric(self, action, value):
        """Validate numeric input"""
        if action == '1':  # Insert
            if not re.match(r'^\d*$', value):
                return False
        return True
    
    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(header_frame, text="Stock Volatility Predictor", style='Header.TLabel').pack(fill=tk.X, ipady=10)
        
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        
        # Controls frame
        controls_frame = ttk.LabelFrame(analysis_frame, text="Analysis Parameters")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Ticker selection
        ttk.Label(controls_frame, text="Select Company:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_var = tk.StringVar()
        self.ticker_cb = ttk.Combobox(controls_frame, textvariable=self.ticker_var, width=30)
        self.ticker_cb['values'] = list(POPULAR_TICKERS.keys())
        self.ticker_cb.current(0)
        self.ticker_cb.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Custom ticker
        ttk.Label(controls_frame, text="Custom Ticker:").grid(row=0, column=2, padx=(20,5), pady=5, sticky="w")
        self.custom_ticker_var = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.custom_ticker_var, width=15).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Date selection
        ttk.Label(controls_frame, text="Start Date:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_var = tk.StringVar(value="2020-01-01")
        ttk.Entry(controls_frame, textvariable=self.start_date_var, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(controls_frame, text="End Date:").grid(row=1, column=2, padx=(20,5), pady=5, sticky="w")
        self.end_date_var = tk.StringVar(value=datetime.date.today().strftime("%Y-%m-%d"))
        ttk.Entry(controls_frame, textvariable=self.end_date_var, width=15).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Volatility window
        ttk.Label(controls_frame, text="Volatility Window (days):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.window_var = tk.IntVar(value=10)
        
        # Add entry for typing
        validate_cmd = (self.root.register(self.validate_numeric), '%d', '%P')
        window_entry = ttk.Entry(controls_frame, textvariable=self.window_var, width=5, 
                                validate="key", validatecommand=validate_cmd)
        window_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Add scale for sliding
        window_scale = ttk.Scale(controls_frame, from_=5, to=30, variable=self.window_var, 
                  orient=tk.HORIZONTAL, length=150)
        window_scale.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        # Add label to show value
        ttk.Label(controls_frame, textvariable=self.window_var).grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        # LSTM parameters
        ttk.Label(controls_frame, text="LSTM Sequence Length:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.seq_length_var = tk.IntVar(value=20)
        ttk.Entry(controls_frame, textvariable=self.seq_length_var, width=8, 
                 validate="key", validatecommand=validate_cmd).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(controls_frame, text="LSTM Epochs:").grid(row=3, column=2, padx=(20,5), pady=5, sticky="w")
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(controls_frame, textvariable=self.epochs_var, width=8, 
                 validate="key", validatecommand=validate_cmd).grid(row=3, column=3, padx=5, pady=5, sticky="w")
        
        # Run button
        run_frame = ttk.Frame(controls_frame)
        run_frame.grid(row=4, column=0, columnspan=4, pady=10)
        self.run_button = ttk.Button(run_frame, text="Run Analysis", command=self.start_analysis_thread, 
                                    style='Run.TButton')
        self.run_button.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        
        # Create main content frame for plots and results
        content_frame = ttk.Frame(analysis_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create plot frame (left side)
        plot_frame = ttk.LabelFrame(content_frame, text="Visualization")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)
        
        # Create figure and canvas
        self.fig = plt.figure(figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create results frame (right side)
        results_frame = ttk.LabelFrame(content_frame, text="Model Performance")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0), pady=5, ipadx=10)
        
        # Create treeview for results
        self.results_tree = ttk.Treeview(
            results_frame, 
            columns=("Metric", "GARCH", "LSTM"), 
            show="headings",
            height=15
        )
        self.results_tree.heading("Metric", text="Metric")
        self.results_tree.heading("GARCH", text="GARCH(1,1)")
        self.results_tree.heading("LSTM", text="LSTM")
        self.results_tree.column("Metric", width=200, anchor="w")
        self.results_tree.column("GARCH", width=120, anchor="center")
        self.results_tree.column("LSTM", width=120, anchor="center")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Add export buttons frame
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=5)
        
        # Logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text.configure(state='disabled', bg='#f8f9fa', font=('Consolas', 9))
        
        # Clear logs button
        clear_frame = ttk.Frame(logs_frame)
        clear_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(clear_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.RIGHT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize analysis thread
        self.analysis_thread = None
        self.stop_analysis = False
    
    def log_message(self, message):
        """Add message to log window"""
        self.log_text.configure(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)  # Auto-scroll to bottom
    
    def clear_logs(self):
        """Clear log window"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
    
    def save_plot(self):
        """Save current plot to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save plot"
        )
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.log_message(f"Plot saved to {file_path}")
                self.status_var.set(f"Plot saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save plot: {str(e)}")
    
    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save results as CSV"
        )
        if file_path:
            try:
                # Create a list of results
                results = []
                for child in self.results_tree.get_children():
                    item = self.results_tree.item(child)
                    results.append(item["values"])
                
                # Create DataFrame and save
                df = pd.DataFrame(results, columns=["Metric", "GARCH(1,1)", "LSTM"])
                df.to_csv(file_path, index=False)
                self.log_message(f"Results exported to {file_path}")
                self.status_var.set(f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def download_stock_data(self, ticker, start_date, end_date):
        """Download stock data from Yahoo Finance"""
        try:
            self.status_var.set(f"📥 Downloading data for {ticker}...")
            self.log_message(f"Downloading stock data for {ticker} from {start_date} to {end_date}")
            self.root.update()
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) == 0:
                raise ValueError(f"No data found for {ticker}")
            self.log_message(f"Downloaded {len(data)} records for {ticker}")
            return data['Close'].dropna()
        except Exception as e:
            raise ValueError(f"Download error: {str(e)}")
    
    def calculate_volatility(self, price_series, window=10):
        """Calculate realized volatility from price series"""
        self.log_message(f"Calculating volatility with {window}-day window...")
        returns = price_series.pct_change().dropna()
        # Annualize volatility
        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        return returns, realized_vol.dropna()
    
    def garch_volatility_forecast(self, returns, volatility, train_ratio=0.8):
        """Forecast volatility using GARCH(1,1) model"""
        # Scale returns for better convergence
        garch_data = returns * 100
        
        # Split train/test
        split_idx = int(len(garch_data) * train_ratio)
        train_garch = garch_data[:split_idx]
        test_garch = garch_data[split_idx:]
        
        # Fit GARCH model
        self.status_var.set("🧮 Training GARCH model...")
        self.log_message("Training GARCH(1,1) model...")
        self.root.update()
        garch_model = arch_model(train_garch, mean='Zero', vol='GARCH', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        
        # Extract parameters
        params = garch_fit.params
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        self.log_message(f"GARCH parameters: ω={omega:.6f}, α={alpha:.6f}, β={beta:.6f}")
        
        # Initialize state with last training values
        last_variance = garch_fit.conditional_volatility[-1] ** 2
        last_return = train_garch.iloc[-1]
        
        # Walk-forward forecast
        self.status_var.set("🔮 Generating GARCH predictions...")
        self.log_message("Generating GARCH predictions...")
        self.root.update()
        test_returns = test_garch.values
        garch_predictions = []
        
        for i in range(len(test_returns)):
            if self.stop_analysis:
                self.log_message("Analysis stopped by user")
                return None, None
                
            # Compute next variance
            next_variance = omega + alpha * last_return**2 + beta * last_variance
            
            # Convert to volatility and scale down
            garch_predictions.append(np.sqrt(next_variance) / 100)
            
            # Update state variables
            last_variance = next_variance
            last_return = test_returns[i]
        
        # Create output series
        garch_predictions = pd.Series(garch_predictions, index=test_garch.index)
        
        # Align with volatility
        common_index = garch_predictions.index.intersection(volatility.index)
        return garch_predictions.loc[common_index], volatility.loc[common_index]
    
    def lstm_volatility_forecast(self, volatility, seq_length=20, epochs=50, train_ratio=0.8):
        """Forecast volatility using LSTM model"""
        # Scale data
        scaler = MinMaxScaler()
        vol_scaled = scaler.fit_transform(volatility.values.reshape(-1, 1))
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(vol_scaled, seq_length)
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        self.status_var.set("🧠 Building LSTM model...")
        self.log_message("Building LSTM model...")
        self.root.update()
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        self.status_var.set(f"⏳ Training LSTM model ({epochs} epochs)...")
        self.log_message(f"Training LSTM model for {epochs} epochs...")
        self.root.update()
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[StopTrainingCallback(self)]
        )
        
        if self.stop_analysis:
            self.log_message("LSTM training stopped by user")
            return None, None
            
        # Make predictions
        self.status_var.set("🔮 Generating LSTM predictions...")
        self.log_message("Generating LSTM predictions...")
        self.root.update()
        lstm_pred = model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
        
        # Create output series
        pred_index = volatility.index[seq_length+split_idx:seq_length+split_idx+len(lstm_pred)]
        lstm_pred = pd.Series(lstm_pred, index=pred_index)
        
        # Align with actual volatility
        actual_vol = volatility.loc[pred_index]
        
        # Log training metrics
        self.log_message(f"LSTM training completed: Final loss = {history.history['loss'][-1]:.6f}")
        
        return lstm_pred, actual_vol
    
    def start_analysis_thread(self):
        """Start analysis in a separate thread"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analysis Running", "Analysis is already in progress!")
            return
            
        self.stop_analysis = False
        self.run_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        # Start analysis in new thread
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Start progress monitor
        self.monitor_progress()
    
    def monitor_progress(self):
        """Monitor analysis progress"""
        if self.analysis_thread.is_alive():
            current_progress = self.progress_var.get()
            if current_progress < 95:
                self.progress_var.set(current_progress + 1)
            self.root.after(100, self.monitor_progress)
        else:
            self.progress_var.set(100)
            self.run_button.config(state=tk.NORMAL)
    
    def run_analysis(self):
        """Main function to run volatility prediction"""
        try:
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Clear plot - completely recreate the figure
            self.fig.clf()  # Clear the entire figure
            
            # Get selected parameters
            selected_company = self.ticker_var.get()
            custom_ticker = self.custom_ticker_var.get().strip()
            
            if custom_ticker:
                ticker = custom_ticker.upper()
            else:
                ticker = POPULAR_TICKERS.get(selected_company, "AAPL")
            
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            window = self.window_var.get()
            seq_length = self.seq_length_var.get()
            epochs = self.epochs_var.get()
            
            # Validate parameters
            if window < 5 or window > 30:
                raise ValueError("Volatility window must be between 5 and 30 days")
            if seq_length < 5 or seq_length > 60:
                raise ValueError("Sequence length must be between 5 and 60")
            if epochs < 10 or epochs > 200:
                raise ValueError("Epochs must be between 10 and 200")
            
            # Download and process data
            prices = self.download_stock_data(ticker, start_date, end_date)
            if self.stop_analysis:
                return
            returns, volatility = self.calculate_volatility(prices, window)
            
            # Run GARCH model
            garch_pred, garch_actual = self.garch_volatility_forecast(returns, volatility)
            if self.stop_analysis or garch_pred is None:
                return
            
            # Run LSTM model
            lstm_pred, lstm_actual = self.lstm_volatility_forecast(volatility, seq_length, epochs)
            if self.stop_analysis or lstm_pred is None:
                return
            
            # Find common period for comparison
            common_index = garch_pred.index.intersection(lstm_pred.index)
            garch_pred = garch_pred.loc[common_index]
            lstm_pred = lstm_pred.loc[common_index]
            actual_vol = volatility.loc[common_index]
            
            # Calculate metrics
            def calculate_metrics(actual, pred):
                return {
                    'MSE': mean_squared_error(actual, pred),
                    'RMSE': np.sqrt(mean_squared_error(actual, pred)),
                    'MAE': mean_absolute_error(actual, pred),
                    'MAPE': np.mean(np.abs((actual - pred) / actual)) * 100
                }
            
            garch_metrics = calculate_metrics(actual_vol, garch_pred)
            lstm_metrics = calculate_metrics(actual_vol, lstm_pred)
            
            # Update results tree
            metrics = [
                ("MSE (Lower is better)", f"{garch_metrics['MSE']:.6f}", f"{lstm_metrics['MSE']:.6f}"),
                ("RMSE (Lower is better)", f"{garch_metrics['RMSE']:.6f}", f"{lstm_metrics['RMSE']:.6f}"),
                ("MAE (Lower is better)", f"{garch_metrics['MAE']:.6f}", f"{lstm_metrics['MAE']:.6f}"),
                ("MAPE % (Lower is better)", f"{garch_metrics['MAPE']:.2f}%", f"{lstm_metrics['MAPE']:.2f}%")
            ]
            
            for metric in metrics:
                self.results_tree.insert("", tk.END, values=metric)
            
            # Create new axes for the plot
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212)
            
            # Price and volatility plot
            ax1.plot(prices, label='Price', color='#1a3d66', linewidth=1.5)
            ax1.set_title(f'{ticker} Stock Price', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.set_facecolor('#f9f9f9')
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(volatility, label='Realized Volatility', color='#e74c3c', alpha=0.8, linewidth=1.5)
            ax1_twin.set_ylabel('Volatility', fontsize=10)
            
            # Combined legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)
            
            # Prediction plot
            ax2.plot(actual_vol, label='Actual Volatility', color='#27ae60', linewidth=2)
            ax2.plot(garch_pred, label='GARCH Prediction', linestyle='--', linewidth=1.5, color='#2980b9')
            ax2.plot(lstm_pred, label='LSTM Prediction', linestyle='--', linewidth=1.5, color='#8e44ad')
            ax2.set_title('Volatility Prediction Comparison', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Annualized Volatility', fontsize=10)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(alpha=0.3, linestyle='--')
            ax2.set_facecolor('#f9f9f9')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update status
            self.status_var.set(f"✅ Analysis complete for {ticker} | {start_date} to {end_date}")
            self.log_message(f"Analysis completed successfully for {ticker}")
            
        except Exception as e:
            self.status_var.set(f"❌ Error: {str(e)}")
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Analysis Error", f"Failed to run analysis:\n{str(e)}")
        finally:
            self.stop_analysis = False

if __name__ == "__main__":
    app = VolatilityPredictor()
