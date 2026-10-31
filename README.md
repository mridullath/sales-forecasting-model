# Retail Sales Forecasting with ARIMA / SARIMA

A time series project that forecasts monthly retail sales using the ARIMA family of models, with a dashboard frontend to explore the results.

---

## 1. What is it?

This project predicts future sales for a retail store from its past sales history. It uses **SARIMA** (Seasonal ARIMA), a classic statistical model for data that has a trend and a repeating seasonal pattern, which is typical of retail sales (for example, peaks in November and December).

The project has two parts:

| Part | File | Purpose |
|---|---|---|
| Forecasting script | `sales_forecasting_arima.py` | The real analysis: loads data, tests it, fits SARIMA, evaluates, forecasts |
| Dashboard preview | `sales_forecast_dashboard.html` | A frontend mockup showing how the results would be presented to a user |

---

## 2. What does it do?

- Converts order-level sales records into **monthly total sales**
- Breaks the series into **trend, seasonality, and noise**
- Checks whether the data is **stationary** (ADF test)
- Automatically searches for the best model settings (lowest AIC)
- Tests the model on held-out months and reports **RMSE, MAE, and MAPE**
- Compares the model against a **seasonal-naive baseline** (same month last year)
- Checks that the model's errors look like random noise (**Ljung-Box test**)
- Forecasts the next **12 months** with **95% confidence intervals**
- Saves charts and a CSV of the forecast

---

## 3. How does it work?

The script follows the standard ARIMA workflow:

1. **Load and aggregate.** Reads `Order Date` and `Sales`, then sums sales by month.
2. **Decompose.** Splits the series into trend, seasonal, and residual parts to see the patterns.
3. **Test stationarity.** ARIMA needs a series with stable statistics over time. The ADF test checks this. If the series is not stationary, it is differenced (this sets `d`, and `D` for seasonal differencing).
4. **Split the data.** The last 6 months are held out for testing. The split is by time, never random.
5. **Select parameters.** A grid search tries combinations of `(p,d,q)(P,D,Q,12)` and keeps the one with the lowest AIC.
   - `p` = autoregressive terms (past values)
   - `d` = number of differences
   - `q` = moving average terms (past errors)
   - `P,D,Q,12` = the same ideas applied to the yearly seasonal cycle
6. **Fit and forecast.** The model is trained on the training set and forecasts the test period.
7. **Evaluate.** Forecasts are compared to actual values using RMSE, MAE, and MAPE, and against the baseline.
8. **Check residuals.** Good residuals look like white noise (Ljung-Box p-value above 0.05).
9. **Forecast the future.** The model is refit on all data and forecasts the next 12 months.

### The dashboard

`sales_forecast_dashboard.html` is a single-page frontend with a sidebar (forecast horizon, test size, auto or manual mode) and two tabs: **Forecast** (key figures, metrics, chart, forecast table, business insight) and **Diagnostics** (residual plot and verdict).

> **Important:** the dashboard is a **preview**. It runs on built-in sample data and uses a simplified seasonal smoothing model in the browser. It does not run the SARIMA script, and its CSV upload box is not connected yet. See "Future improvements" for how to make it fully functional.

---

## 4. How to run it

### Requirements

- Python 3.9 or newer
- Libraries: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`

```
pip install pandas numpy matplotlib statsmodels scikit-learn
```

### Steps

1. Put `sales_forecasting_arima.py` in a folder.
2. Add your dataset as `Superstore.csv` in the same folder (see section 5). If the file is missing, the script runs on synthetic demo data.
3. Run:
   ```
   python sales_forecasting_arima.py
   ```
4. Read the console output (best model, metrics, forecast table). The script also creates:
   - `01_decomposition.png`
   - `02_residual_diagnostics.png`
   - `03_test_forecast.png`
   - `04_future_forecast.png`
   - `future_sales_forecast.csv`

To view the dashboard, open `sales_forecast_dashboard.html` in any modern browser.

---

## 5. How to add a dataset

### Recommended dataset

The Superstore Sales dataset from Kaggle:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

A free Kaggle account is needed to download it.

### Steps

1. Download and unzip the file.
2. Rename the CSV to `Superstore.csv` and place it next to the script.
3. Confirm the file has columns named **`Order Date`** and **`Sales`**.
4. Run the script.

### Using your own data

Any sales CSV works if it has a date column and a sales amount column.

- **Different column names:** edit the `load_monthly_sales` function and change `"Order Date"` and `"Sales"` to your column names.
- **Different file name:** change the `CSV_PATH` value at the top of the script.
- **Weird characters or encoding errors:** the script reads files with `latin-1` encoding. Change it to `utf-8` if that suits your file.
- **Dates parsed wrongly:** specify the format, for example `pd.to_datetime(df["Order Date"], format="%d/%m/%Y")`.
- **Weekly or daily data:** change the resampling rule (`"MS"` is month start) and set `SEASONAL_PERIOD` to match (52 for weekly, 7 for daily weekly-patterns).

### Data tips

- Use **at least 3 to 4 years** of monthly data. Seasonal models need several full cycles.
- Fill or investigate missing months before running.
- Remove obvious errors or one-off outliers, or at least note them.

---

## 6. Real-world business uses

| Area | How forecasting helps |
|---|---|
| **Inventory management** | Order the right stock ahead of peak months and avoid overstocking in slow ones |
| **Staffing** | Plan seasonal hiring and shift schedules around expected demand |
| **Budgeting and finance** | Build revenue projections and cash flow plans |
| **Sales targets** | Set realistic, data-backed targets for teams and regions |
| **Marketing** | Time campaigns and promotions around predicted demand dips and peaks |
| **Supply chain** | Coordinate with suppliers and logistics on expected volumes |
| **Anomaly detection** | Flag months where actual sales deviate sharply from the forecast |

The same method applies beyond retail: electricity demand, website traffic, air quality, agricultural prices, call-centre volumes, and more.

---

## 7. Limitations

- **Short history.** With only about 4 years of data, confidence intervals are wide and the seasonal pattern is estimated from few cycles.
- **Sales history only.** The model cannot see promotions, holidays, price changes, competitors, or economic conditions.
- **Assumes the past repeats.** Sudden shocks (a pandemic, a new competitor) are not predicted.
- **One series at a time.** The base version forecasts total sales, not each category or region.
- **Dashboard is a preview.** It uses sample data and a simplified model (see section 3).

---

## 8. Future improvements

**Modelling**
- Add **SARIMAX** with external variables (discounts, holidays, promotions)
- Compare against **Holt-Winters, Prophet, and LSTM** and report which wins
- Use **`pmdarima.auto_arima`** for faster, smarter parameter search
- Apply **rolling-origin cross-validation** for more reliable error estimates
- Forecast by **category, region, or segment**

**Dashboard**
- Make the **CSV upload work**: read the file in the browser, aggregate by month, and let the user pick the date and sales columns
- Build the full app in **Streamlit** so it runs the real SARIMA code
- Add **download buttons** for the forecast table and charts
- Add a **what-if panel** (for example, "sales grow 10% more than forecast")
- Auto-generate a written **business insights** summary

**Deployment**
- Deploy on **Streamlit Community Cloud** for a shareable link
- Schedule automatic **monthly retraining** as new data arrives
- Add **alerts** when actual sales drift far from the forecast

---

## 9. Tech stack

Python, pandas, NumPy, statsmodels, scikit-learn, Matplotlib. The dashboard is plain HTML, CSS, and JavaScript with no external libraries.

---

## 10. Project structure

```
project/
├── sales_forecasting_arima.py     # forecasting script
├── sales_forecast_dashboard.html  # dashboard preview
├── Superstore.csv                 # your dataset (add this)
├── README.md
└── (generated)  01_decomposition.png, 02_residual_diagnostics.png,
                 03_test_forecast.png, 04_future_forecast.png,
                 future_sales_forecast.csv
```

---

## 11. Metrics glossary

- **RMSE:** average error size, penalizing large misses more. Lower is better.
- **MAE:** average absolute error in the same units as sales.
- **MAPE:** average error as a percentage of actual sales. Easy to explain to non-technical readers.
- **AIC:** model quality score that rewards fit but penalizes complexity. Lower is better.
- **Seasonal-naive baseline:** forecasting each month as the same month last year. A useful model must beat it.
