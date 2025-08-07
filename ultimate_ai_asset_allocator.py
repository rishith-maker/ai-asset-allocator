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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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

# ============ CONFIG ===================
# Setup Pytrends with retries and headers
requests_args = {
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
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
    "ea6d44c6e1de4e64823c4d372384adb5",
    "djouQaN6E7XYWmXQ9tCGXcjWgdHFDC29"
]

def get_fred_api_key():
    return random.choice(FRED_API_KEYS)

# 🗂 Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load FinBERT sentiment model once
finbert = hf_pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# 🚀 Create FastAPI app
app = FastAPI(title="AI IPO Hedge Fund-Grade Pipeline")

# ============ MODELS & API =============
class PredictRequest(BaseModel):
    tickers: List[str]
    start_date: str = "2021-01-01"
    end_date: str = None

# ============ DATA LOADERS =============
def get_price_data(tickers, start='2021-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.dropna()

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
    features = pd.concat([returns.shift(1), momentum.shift(1), volatility.shift(1)], axis=1)
    return features.dropna(), returns.loc[features.dropna().index]

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

# ============ MODEL TRAINING + SWITCHING ============
def train_model(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model

def train_all_models(features, target, tickers):
    models = {}
    predictions = pd.DataFrame(index=features.index)
    for ticker in tickers:
        y = target[ticker].dropna()
        X = features.loc[y.index]
        X = calculate_vif(X)
        X = apply_rfe(X, y, min(10, X.shape[1]))
        model = train_model(X, y)
        models[ticker] = model
        predictions[ticker] = model.predict(features[X.columns])
    return models, predictions
   # ============ PORTFOLIO OPTIMIZATION =============
   def optimize_portfolio(predicted_returns):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(predicted_returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    allocation = pd.Series(optimal_weights, index=mean_returns.index)
    return allocation

# ✅ STEP 3: Run optimizer and debug print
print("✅ Running portfolio optimizer...")
allocation = optimize_portfolio(predicted_returns)
print("✅ Allocation generated:\n", allocation)

# ✅ STEP 4: Create results folder and save allocation
os.makedirs("results", exist_ok=True)

print("✅ Saving allocation to results/allocation.csv")
allocation.to_csv("results/allocation.csv")



# ============ BACKTESTING =====================
def run_backtest(predicted_returns, weights):
    import bt
    price_idx = (1 + predicted_returns.fillna(0)).cumprod()
    def strategy(prices): return weights
    strat = bt.Strategy("AI_Optimized", [bt.algos.RunMonthly(), bt.algos.SelectAll(), bt.algos.WeighTarget(strategy), bt.algos.Rebalance()])
    test = bt.Backtest(strat, price_idx)
    res = bt.run(test)
    res.plot(title="Equity Curve")
    plt.savefig(f"{RESULTS_DIR}/backtest_equity_curve.png")
    return res

# ============ DRIFT MONITOR ====================
def run_drift_monitor(train_feats, live_feats):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_feats, current_data=live_feats)
    report.save_html(f"{RESULTS_DIR}/drift_report.html")

# ============ API ENDPOINT =====================
@app.post("/predict")
def predict_allocation(request: PredictRequest):
    tickers = request.tickers
    prices = get_price_data(tickers, request.start_date, request.end_date)
    features, target = engineer_features(prices)
    macro = get_macro_data(request.start_date, request.end_date)
    features = merge_macro_features(features, macro)
    regimes = detect_market_regimes(prices)

    mock_filings = {ticker: f"{ticker} is a growth tech firm expanding rapidly..." for ticker in tickers}
    sentiment = get_sentiment_scores(mock_filings)
    for col in sentiment.columns:
        for ticker in sentiment.index:
            features.loc[:, f"{ticker}_{col}"] = sentiment.loc[ticker, col]

    models, preds = train_all_models(features, target, tickers)
    allocation = optimize_portfolio(preds)
    joblib.dump(models, f"{RESULTS_DIR}/models.pkl")
    allocation.to_csv(f"{RESULTS_DIR}/allocation.csv")
    run_backtest(preds, allocation)
    run_drift_monitor(features, features.tail(50))

    return {
        "tickers": tickers,
        "allocation": allocation.to_dict(),
        "message": "Complete pipeline executed with regime, sentiment, macro, drift, and optimization."
    }
def main(tickers=None, capital=10000, start='2021-01-01', end=None):
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
    print("🔍 Fetching IPO price data...")
    price_data = get_ipo_data(tickers, start=start, end=end)

    print("📊 Training return prediction models...")
    models, predicted_returns = train_return_model(price_data)

    print("📈 Optimizing portfolio using Sharpe Ratio...")
    allocation = optimize_portfolio(predicted_returns)

    print("💾 Exporting results to CSV and chart...")
    export_results(allocation, capital=capital)

    print("✅ Pipeline complete.")
    