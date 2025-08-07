# === AI Financial Pipeline: Clean Merged Version with Dynamic Risk-Free Rate ===

import os
import datetime
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import bt

# Optional: FRED + Drift + Sentiment imports (can be configured later)
from fredapi import Fred
from transformers import pipeline as hf_pipeline
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

warnings.simplefilter(action='ignore', category=FutureWarning)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# === Risk-Free Rate Data ===
def get_risk_free_rate(start='2021-01-01', end=None):
    """
    Fetch the 10-year Treasury yield (DGS10) from FRED.
    Falls back to 3-month Treasury bill (DTB3) if DGS10 is unavailable.
    Returns annual rate in decimal form.
    """
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    fred = Fred(api_key="YOUR_KEY_HERE")
 # assumes FRED_API_KEY env var is set
    try:
        # DGS10 is daily 10-year yield in percent
        series = fred.get_series('DGS10', observation_start=start, observation_end=end)
        if not series.empty:
            rf = series.iloc[-1] / 100.0
            print(f"📊 Using FRED DGS10 (10-yr) rate: {rf:.4f} ({rf*100:.2f}%)")
            return rf
    except Exception as e:
        print(f"⚠️  Failed to fetch DGS10: {e}")
    try:
        series = fred.get_series('DTB3', observation_start=start, observation_end=end)
        if not series.empty:
            rf = series.iloc[-1] / 100.0
            print(f"📊 Using FRED DTB3 (3-mo) rate: {rf:.4f} ({rf*100:.2f}%)")
            return rf
    except Exception as e:
        print(f"⚠️  Failed to fetch DTB3: {e}")
    print("⚠️  Could not fetch FRED rates, using default 2.5%")
    return 0.025



# === Price Data ===
def get_price_data(tickers, start='2021-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by='ticker')
    price_data = {}
    if len(tickers) == 1:
        df = data
        price_data[tickers[0]] = df['Adj Close'].dropna() if 'Adj Close' in df.columns else df['Close'].dropna()
    else:
        for ticker in tickers:
            df = data[ticker]
            price_data[ticker] = df['Adj Close'].dropna() if 'Adj Close' in df.columns else df['Close'].dropna()
    return price_data


# === Feature Engineering ===
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


# === Model Training ===
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# === Portfolio Optimization ===
def optimize_portfolio(predicted_returns, cov_matrix, tickers, rf_rate=0.025):
    """
    Optimize portfolio using Sharpe ratio with dynamic risk-free rate.
    """
    mu = np.array([predicted_returns[t] for t in tickers])
    cov = cov_matrix.loc[tickers, tickers].values

    def neg_sharpe(weights):
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -(port_return - rf_rate) / port_vol if port_vol != 0 else 1e6

    x0 = np.array([1 / len(tickers)] * len(tickers))
    bounds = [(0, 1)] * len(tickers)
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    return {t: w for t, w in zip(tickers, result.x)} if result.success else {t: 1/len(tickers) for t in tickers}


# === Backtest ===
def run_backtest(price_data, weights):
    price_df = pd.DataFrame(price_data).dropna()
    strat = bt.Strategy("AI_Optimized", [
        bt.algos.RunMonthly(),
        bt.algos.SelectAll(),
        bt.algos.WeighSpecified(**weights),
        bt.algos.Rebalance()
    ])
    test = bt.Backtest(strat, price_df)
    res = bt.run(test)
    res.plot(title="Equity Curve")
    plt.savefig(f"{RESULTS_DIR}/backtest_equity_curve.png")
    return res


# === Metrics ===
def calculate_sharpe_ratio(daily_returns, rf_rate=0.025):
    """
    Calculate Sharpe ratio using actual risk-free rate.
    """
    daily_rf = rf_rate / 252
    excess_returns = daily_returns - daily_rf
    if excess_returns.std() != 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    return sharpe


def calculate_annual_return(equity_curve):
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    return (1 + total_return) ** (252 / len(equity_curve)) - 1


# === Drift Monitor ===
def run_drift_monitor(train_feats, live_feats):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_feats, current_data=live_feats)
    report.save_html(f"{RESULTS_DIR}/drift_report.html")


# === Export Results ===
def export_results(allocation, rf_rate=None):
    results_df = pd.DataFrame({
        'ticker': list(allocation.keys()),
        'weight': list(allocation.values())
    })
    if rf_rate is not None:
        results_df['risk_free_rate'] = rf_rate
    results_df.to_csv(f"{RESULTS_DIR}/allocation.csv", index=False)


# === Main ===
def main():
    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
    print("🔍 Fetching data...")
    price_data = get_price_data(tickers)

    # Fetch current risk-free rate
    rf_rate = get_risk_free_rate()

    predicted_returns = {}
    print("📊 Training models...")
    for ticker in tickers:
        feats, targets = engineer_features(price_data[ticker])
        model, X_test, y_test = train_model(feats, targets)
        preds = model.predict(X_test)
        predicted_returns[ticker] = preds.mean()
        print(f"{ticker} RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

    print("📈 Optimizing portfolio...")
    combined = pd.concat(price_data.values(), axis=1)
    combined.columns = tickers
    cov_matrix = combined.pct_change().cov()
    allocation = optimize_portfolio(predicted_returns, cov_matrix, tickers, rf_rate)
    export_results(allocation, rf_rate)

    print("🔄 Backtesting...")
    backtest = run_backtest(price_data, allocation)
    equity = backtest.prices["AI_Optimized"]
    equity.to_csv(f"{RESULTS_DIR}/equity_curve.csv")

    # Use actual risk-free rate in Sharpe calculation
    sharpe = calculate_sharpe_ratio(equity.pct_change().dropna(), rf_rate)
    ann_return = calculate_annual_return(equity)
    print(f"📈 Sharpe Ratio: {sharpe:.4f} | Annual Return: {ann_return:.2%}")
    print(f"📊 Risk-Free Rate Used: {rf_rate:.4f} ({rf_rate*100:.2f}%)")

    print("📊 Drift analysis...")
    f1, _ = engineer_features(price_data[tickers[0]])
    f2, _ = engineer_features(price_data[tickers[-1]])
    f1.index.name = "index"
    f2.index.name = "index"
    run_drift_monitor(f1, f2)

    print("✅ Done.")


if __name__ == "__main__":
    main()

