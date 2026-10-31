"""
Retail Sales Forecasting with ARIMA / SARIMA
--------------------------------------------
Dataset: Superstore (Kaggle) -> needs columns 'Order Date' and 'Sales'.
Place the CSV next to this script as 'Superstore.csv'.
If the file is missing, synthetic monthly data is generated so the script still runs.

Install:  pip install pandas numpy matplotlib statsmodels scikit-learn
"""

import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up in a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

CSV_PATH = "Superstore.csv"
SEASONAL_PERIOD = 12      # monthly data -> yearly seasonality
TEST_MONTHS = 6           # hold-out period for evaluation
FUTURE_MONTHS = 12        # how far ahead to forecast at the end


# ---------------------------------------------------------------- 1. LOAD DATA
def load_monthly_sales(path):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        series = df.set_index("Order Date")["Sales"].resample("MS").sum()
    else:
        print(f"'{path}' not found -> using synthetic data for demo.\n")
        rng = np.random.default_rng(42)
        idx = pd.date_range("2014-01-01", periods=48, freq="MS")
        trend = np.linspace(20000, 45000, 48)
        season = 8000 * np.sin(2 * np.pi * (idx.month - 3) / 12)
        series = pd.Series(trend + season + rng.normal(0, 2500, 48), index=idx)
    series.name = "Sales"
    return series


sales = load_monthly_sales(CSV_PATH)
print("Monthly series:", sales.index.min().date(), "to", sales.index.max().date(),
      f"({len(sales)} points)")
print(sales.head(), "\n")


# ---------------------------------------------------------------- 2. EXPLORE
decomp = seasonal_decompose(sales, model="additive", period=SEASONAL_PERIOD)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig("01_decomposition.png", dpi=120)
plt.close()


# ---------------------------------------------------------------- 3. STATIONARITY
def adf_report(x, label):
    stat, p, *_ = adfuller(x.dropna())
    verdict = "stationary" if p < 0.05 else "NON-stationary"
    print(f"ADF [{label}]: statistic={stat:.3f}, p-value={p:.4f} -> {verdict}")
    return p


p_raw = adf_report(sales, "original")
p_diff = adf_report(sales.diff(), "1st difference")
p_sdiff = adf_report(sales.diff(SEASONAL_PERIOD), "seasonal difference")
print()


# ---------------------------------------------------------------- 4. TRAIN/TEST
train, test = sales.iloc[:-TEST_MONTHS], sales.iloc[-TEST_MONTHS:]


# ---------------------------------------------------------------- 5. MODEL SELECTION
# Grid search over (p,d,q)(P,D,Q,s), pick the lowest AIC.
p_range = q_range = range(0, 3)
P_range = Q_range = range(0, 2)
d, D = 1, 1  # from the stationarity tests above; adjust if your data differs

best = {"aic": np.inf}
for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
    try:
        fit = SARIMAX(
            train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        if fit.aic < best["aic"]:
            best = {"aic": fit.aic, "order": (p, d, q), "seasonal": (P, D, Q, SEASONAL_PERIOD)}
    except Exception:
        continue

print("Best model by AIC:", best, "\n")


# ---------------------------------------------------------------- 6. FIT + FORECAST TEST PERIOD
model = SARIMAX(
    train,
    order=best["order"],
    seasonal_order=best["seasonal"],
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)
print(model.summary().tables[1], "\n")

fc = model.get_forecast(steps=TEST_MONTHS)
pred = fc.predicted_mean
ci = fc.conf_int()


# ---------------------------------------------------------------- 7. EVALUATE
rmse = np.sqrt(mean_squared_error(test, pred))
mae = mean_absolute_error(test, pred)
mape = np.mean(np.abs((test - pred) / test)) * 100
print(f"RMSE: {rmse:,.2f}\nMAE : {mae:,.2f}\nMAPE: {mape:.2f}%\n")

# Naive baseline: repeat the same month from last year
naive = sales.shift(SEASONAL_PERIOD).loc[test.index]
naive_mape = np.mean(np.abs((test - naive) / test)) * 100
print(f"Seasonal-naive baseline MAPE: {naive_mape:.2f}%  (ARIMA should beat this)\n")


# ---------------------------------------------------------------- 8. RESIDUAL CHECK
lb = acorr_ljungbox(model.resid.iloc[SEASONAL_PERIOD + 1:], lags=[10], return_df=True)
print("Ljung-Box test (p > 0.05 means residuals look like white noise):")
print(lb, "\n")
try:
    model.plot_diagnostics(figsize=(10, 8), lags=8)
    plt.tight_layout()
    plt.savefig("02_residual_diagnostics.png", dpi=120)
    plt.close()
except ValueError:
    # Short series: fall back to a simple residual plot
    resid = model.resid.iloc[SEASONAL_PERIOD + 1:]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(resid); ax[0].set_title("Residuals over time")
    ax[1].hist(resid, bins=10); ax[1].set_title("Residual distribution")
    plt.tight_layout()
    plt.savefig("02_residual_diagnostics.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------- 9. PLOT TEST FORECAST
plt.figure(figsize=(11, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Actual", color="black")
plt.plot(pred, label="Forecast", color="red")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.15)
plt.title("Hold-out forecast vs actual sales")
plt.legend()
plt.tight_layout()
plt.savefig("03_test_forecast.png", dpi=120)
plt.close()


# ---------------------------------------------------------------- 10. FUTURE FORECAST
final = SARIMAX(
    sales,
    order=best["order"],
    seasonal_order=best["seasonal"],
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

future = final.get_forecast(steps=FUTURE_MONTHS)
future_df = pd.DataFrame({
    "forecast": future.predicted_mean,
    "lower_95": future.conf_int().iloc[:, 0],
    "upper_95": future.conf_int().iloc[:, 1],
}).round(2)
future_df.to_csv("future_sales_forecast.csv")
print(f"Next {FUTURE_MONTHS} months forecast:\n", future_df, "\n")

plt.figure(figsize=(11, 5))
plt.plot(sales, label="History")
plt.plot(future_df["forecast"], label="Forecast", color="red")
plt.fill_between(future_df.index, future_df["lower_95"], future_df["upper_95"],
                 color="red", alpha=0.15)
plt.title(f"Sales forecast for the next {FUTURE_MONTHS} months")
plt.legend()
plt.tight_layout()
plt.savefig("04_future_forecast.png", dpi=120)
plt.close()

print("Saved: 01_decomposition.png, 02_residual_diagnostics.png, "
      "03_test_forecast.png, 04_future_forecast.png, future_sales_forecast.csv")
