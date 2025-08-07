import pandas as pd
import numpy as np
import datetime
import os
import yfinance as yf
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from fredapi import Fred
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from transformers import pipeline as hf_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from pytrends.request import TrendReq
from scipy.optimize import minimize
import bt
import random
import time
import uuid
import warnings
from sklearn.model_selection import train_test_split  # Add this line
from sklearn.metrics import mean_squared_error  # Ensure this is also imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Your function or script code follows


# ✅ Define the function FIRST
def get_price_data(tickers, start='2021-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by='ticker')
        price_data = {}

        if len(tickers) == 1:
            df = data
            if 'Adj Close' in df.columns:
                price_data[tickers[0]] = df['Adj Close'].dropna()
            else:
                price_data[tickers[0]] = df['Close'].dropna()
        else:
            for ticker in tickers:
                df = data[ticker]
                if 'Adj Close' in df.columns:
                    price_data[ticker] = df['Adj Close'].dropna()
                else:
                    price_data[ticker] = df['Close'].dropna()

        return price_data
    except Exception as e:
        print(f"An error occurred while fetching price data: {e}")
        return None


# ...existing code...
# ✅ Define the function FIRST
def get_price_data(tickers, start='2021-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by='ticker')
        price_data = {}

        if len(tickers) == 1:
            df = data
            if 'Adj Close' in df.columns:
                price_data[tickers[0]] = df['Adj Close'].dropna()
            else:
                price_data[tickers[0]] = df['Close'].dropna()
        else:
            for ticker in tickers:
                df = data[ticker]
                if 'Adj Close' in df.columns:
                    price_data[ticker] = df['Adj Close'].dropna()
                else:
                    price_data[ticker] = df['Close'].dropna()

        return price_data
    except Exception as e:
        print(f"An error occurred while fetching price data: {e}")
        return None

# Remove the second get_price_data definition!
# ...existing code...

# ============ CONFIG ===================
# Setup Pytrends with retries and headers
requests_args = {
    'headers': {
        'User -Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
}
pytrends = TrendReq(
    hl='en-US',
    tz=360,
    retries=3,
    backoff_factor=1,
    requests_args=requests_args
)
time.sleep(10)

# 🔑 Store your 3 API keys here
FRED_API_KEYS = [
    "78bec6e7bd0c1934652e866d6da6dace",
   
]

def get_fred_api_key():
    return random.choice(FRED_API_KEYS)

# 🗂 Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load FinBERT sentiment model once
finbert = hf_pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")




# ============ DATA LOADERS =============
def get_macro_data(start, end):
    fred = Fred(api_key=get_fred_api_key())
    series = {
        'FEDFUNDS': 'FedFundsRate',
        'CPIAUCSL': 'InflationCPI',
        'UNRATE': 'UnemploymentRate'
    }
    df = pd.DataFrame()
    for code, name in series.items():
        df[name] = fred.get_series(code, observation_start=start, observation_end=end)
    df.index = pd.to_datetime(df.index)
    return df.resample('D').ffill().fillna(method='bfill')

# ============ FEATURE ENGINEERING =============
def engineer_features(prices):
    returns = prices.pct_change().dropna()
    momentum = returns.rolling(window=5).mean()
    volatility = returns.rolling(window=5).std()
    features = pd.DataFrame({
        'returns': returns.shift(1),
        'momentum': momentum.shift(1),
        'volatility': volatility.shift(1)
    })
    features = features.dropna()
    return features, returns.loc[features.index]

def merge_macro_features(price_feats, macro_df):
    return pd.concat([price_feats, macro_df.reindex(price_feats.index, method='ffill')], axis=1)

# ============ FEATURE SELECTION =============
def calculate_vif(X, thresh=5.0):
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    selected = vif_df[vif_df["VIF"] < thresh]["feature"].tolist()
    return X[selected]

def apply_rfe(X, y, n=10):
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n)
    selector.fit(X, y)
    return X[X.columns[selector.support_]]

# ============ SENTIMENT PIPELINE =============
def get_sentiment_scores(text_dict):
    out = {}
    for ticker, text in text_dict.items():
        result = finbert(text[:512])[0]
        score = {'positive': 0, 'neutral': 0, 'negative': 0}
        score[result['label'].lower()] = result['score']
        out[ticker] = score
    return pd.DataFrame(out).T

# ============ REGIME DETECTOR =============
def detect_market_regimes(prices, window=20, threshold=0.02):
    vol = prices.pct_change().rolling(window).std()
    regimes = (vol > threshold).astype(str).replace({'True': 'high_vol', 'False': 'low_vol'})
    return regimes

def prepare_features_targets(price_series):
    returns = price_series.pct_change().dropna()
    X = returns.shift(1).dropna()  # lagged returns as features (example)
    y = returns.loc[X.index]
    return X, y


# ============ MODEL TRAINING + SWITCHING ============
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def optimize_portfolio(predicted_returns, cov_matrix=None, tickers=None):
    """
    Optimize portfolio allocation based on predicted returns and covariance matrix.
    Returns a dictionary of allocations.
    """
    allocation = {}
    if predicted_returns is None or not predicted_returns or cov_matrix is None:
        print("❌ No predicted returns or covariance matrix available for portfolio optimization.")
        return allocation
    if tickers is None:
        tickers = list(predicted_returns.keys())

    mu = np.array([predicted_returns[ticker] for ticker in tickers])
    cov = cov_matrix.loc[tickers, tickers].values

    def neg_sharpe(weights):
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        # Avoid division by zero
        if port_vol == 0:
            return 1e6
        return -port_return / port_vol

    # Constraints: weights sum to 1, weights >= 0
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in tickers]
    x0 = np.array([1 / len(tickers)] * len(tickers))

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    if not result.success:
        print("⚠️ Optimization failed, using equal weights.")
        allocation = {ticker: 1 / len(tickers) for ticker in tickers}
    else:
        allocation = {ticker: weight for ticker, weight in zip(tickers, result.x)}
    return allocation
# ============ BACKTESTING =====================
def run_backtest(price_data, weights):
    if price_data is None or weights is None:
        print("❌ Invalid input for backtesting.")
        return None

    # Remove tickers with empty price series
    valid_price_data = {k: v for k, v in price_data.items() if isinstance(v, pd.Series) and not v.empty}
    if not valid_price_data:
        print("❌ No valid price data for backtesting.")
        return None

    price_df = pd.DataFrame(valid_price_data)
    price_df = price_df.dropna()

    # Filter weights to match available tickers
    valid_weights = {k: weights[k] for k in price_df.columns if k in weights}

    if price_df.empty or not valid_weights:
        print("❌ No valid price data or weights for backtesting.")
        return None

    strat = bt.Strategy(
        "AI_Optimized",
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectAll(),
            bt.algos.WeighSpecified(**valid_weights),
            bt.algos.Rebalance()
        ]
    )
    test = bt.Backtest(strat, price_df)
    res = bt.run(test)
    res.plot(title="Equity Curve")
    plt.savefig(f"{RESULTS_DIR}/backtest_equity_curve.png")
    return res

# ============ DRIFT MONITOR ====================
def run_drift_monitor(train_feats, live_feats):
    if train_feats is None or live_feats is None or train_feats.empty or live_feats.empty:
        print("❌ Invalid input for drift monitoring.")
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_feats, current_data=live_feats)
    report.save_html(f"{RESULTS_DIR}/drift_report.html")



def export_results(allocation, capital=10000):
    """Save allocation results to CSV and generate plot"""
    print("DEBUG: allocation =", allocation)

    # Use Series for scalar dict
    pd.Series(allocation).to_csv(f"{RESULTS_DIR}/allocation.csv")

    print(f"✅ Saved allocation results to {RESULTS_DIR}/allocation.csv")


def main():
    print("🔍 Fetching price data...")

    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
    capital = 10000
    predicted_returns = {}

    # Fetch price data
    try:
        price_data = get_price_data(tickers)
        if price_data is None or any(data is None for data in price_data.values()):
            raise ValueError("Some price data could not be fetched.")
    except Exception as e:
        print(f"❌ Error fetching price data: {e}")
        return

    print("📊 Training return prediction models...")
    # ... Sentiment, Macro, Regime steps ...

    # Train model per ticker and predict returns
    for ticker in tickers:
        try:
            X, y = prepare_features_targets(price_data[ticker])
            model, X_test, y_test = train_model(X, y)
            predictions = model.predict(X_test)
            predicted_returns[ticker] = predictions.mean()
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            print(f"{ticker} RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"⚠️ Error training model for {ticker}: {e}")
            predicted_returns[ticker] = 0.0

    # Calculate covariance matrix for portfolio optimization
    try:
        combined_df = pd.concat(price_data.values(), axis=1)
        combined_df.columns = tickers
        cov_matrix = combined_df.pct_change().cov()
    except Exception as e:
        print(f"❌ Error computing covariance matrix: {e}")
        return



    print("📈 Optimizing portfolio...")
    allocation = optimize_portfolio(predicted_returns, cov_matrix, tickers)

    if not allocation or not any(allocation.values()):
        print("⚠️ Allocation is empty. Using equal weights as fallback.")
        allocation = {ticker: 1 / len(tickers) for ticker in tickers}

    print("💾 Exporting results...")
    export_results(allocation, capital=capital)

    print("🔄 Running backtest...")
    run_backtest(price_data, allocation)
    print("✅ Backtest complete. Equity curve saved.")

    print("🔎 Running drift monitor...")
    train_feats, _ = engineer_features(price_data[tickers[0]])
    live_feats, _ = engineer_features(price_data[tickers[-1]])
    run_drift_monitor(train_feats, live_feats)
    print("✅ Drift report saved.")

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()