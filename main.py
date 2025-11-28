import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import joblib

BASE = "USD"
QUOTE = "EUR"
START_DATE = "2010-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
SAVE_MODEL_PATH = "models"
np.random.seed(42)

def fetch_exchangerate_host(base, quote, start, end):
    url = "https://api.exchangerate.host/timeseries"
    params = {"start_date": start, "end_date": end, "base": base, "symbols": quote, "places": 6}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True):
        raise RuntimeError("Помилка API: " + str(data))
    rows = [{"date": d, "rate": v.get(quote, None)} for d, v in data["rates"].items()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date").asfreq("D")
    return df

def fetch_frankfurter(base, quote, start, end):
    url = f"https://api.frankfurter.app/{start}..{end}"
    params = {"from": base, "to": quote}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = [{"date": d, "rate": v.get(quote, None)} for d, v in data.get("rates", {}).items()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date").asfreq("D")
    return df

def build_dataset():
    try:
        df = fetch_exchangerate_host(BASE, QUOTE, START_DATE, END_DATE)
        source = "exchangerate.host"
    except Exception as e:
        print("exchangerate.host недоступний:", e, "-> пробуємо frankfurter")
        df = fetch_frankfurter(BASE, QUOTE, START_DATE, END_DATE)
        source = "frankfurter"
    print(f"Завантажено {len(df)} рядків з {source}")

    missing_before = df['rate'].isna().sum()
    df['rate'] = df['rate'].interpolate(method='time').ffill().bfill()
    missing_after = df['rate'].isna().sum()
    print(f"Пропусків до обробки: {missing_before}, після інтерполяції: {missing_after}")

    df['pct_change'] = df['rate'].pct_change().fillna(0)
    z = (df['pct_change'] - df['pct_change'].mean()) / (df['pct_change'].std() + 1e-9)
    anomalies = df[np.abs(z) > 3.5]
    print(f"Виявлено аномалій: {len(anomalies)}")
    df.loc[np.abs(z) > 3.5, 'rate'] = np.nan
    df['rate'] = df['rate'].interpolate(method='time').ffill().bfill()
    df.drop(columns=['pct_change'], inplace=True)

    df_feat = df.copy()
    for lag in [1,2,3,7,14,30]:
        df_feat[f'lag_{lag}'] = df_feat['rate'].shift(lag)
    df_feat['rolling_7'] = df_feat['rate'].rolling(7).mean().shift(1)
    df_feat['rolling_30'] = df_feat['rate'].rolling(30).mean().shift(1)
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat = df_feat.dropna()
    return df, df_feat

def plot_decomposition(series, model='additive', period=365):
    res = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
    fig, axes = plt.subplots(4,1, figsize=(12,9), sharex=True)
    res.observed.plot(ax=axes[0], title='Спостереження')
    res.trend.plot(ax=axes[1], title='Тренд')
    res.seasonal.plot(ax=axes[2], title='Сезонність')
    res.resid.plot(ax=axes[3], title='Залишки')
    plt.tight_layout()
    plt.show()
    plt.close()

def train_sarimax(series, train_end_date):
    s = series.asfreq('D').ffill().bfill()
    if isinstance(train_end_date, str):
        train_end_date = pd.to_datetime(train_end_date)
    train = s[:train_end_date]
    test = s[train_end_date + pd.Timedelta(days=1):]

    print(f"Навчання на {len(train)} днях, тестування на {len(test)} днях")
    print("Підбір параметрів Auto ARIMA...")

    arima = pm.auto_arima(train, seasonal=True, m=7, stepwise=True, suppress_warnings=True,
                          error_action='ignore', max_p=3, max_q=3, max_P=2, max_Q=2)
    print("Вибрані порядки:", arima.order, arima.seasonal_order)

    model = sm.tsa.statespace.SARIMAX(train, order=arima.order,
                                      seasonal_order=arima.seasonal_order,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    fitted_model = model.fit(disp=False)

    n_forecast = len(test)
    pred = fitted_model.get_forecast(steps=n_forecast)
    pred_mean = pd.Series(pred.predicted_mean, index=test.index)

    mae = mean_absolute_error(test, pred_mean)
    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    print(f"SARIMAX MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    return fitted_model, pred_mean, test, (mae, rmse)

def train_feature_model(df_feat, target_col='rate', train_end_date=None):
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    if train_end_date is not None:
        if isinstance(train_end_date, str):
            train_end_date = pd.to_datetime(train_end_date)
        split_idx = X.index.get_loc(train_end_date) + 1
    else:
        split_idx = int(len(X) * 0.85)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"RandomForest MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    return model, pd.Series(pred, index=y_test.index), y_test, (mae, rmse)

def plot_actual_pred(actual, pred, title="Фактичне vs Прогнозоване", mae=None, rmse=None):
    plt.figure(figsize=(12,5))
    plt.plot(actual.index, actual.values, label='Фактичне', linewidth=1.5)
    plt.plot(pred.index, pred.values, label='Прогнозоване', linewidth=1.5)
    plt.legend()
    if mae is not None and rmse is not None:
        plt.title(f"{title}\nMAE: {mae:.6f}, RMSE: {rmse:.6f}")
    else:
        plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel(f"Курс ({BASE}/{QUOTE})")
    plt.grid(alpha=0.3)
    plt.show()
    plt.close()

def plot_plotly(actual, pred, title="Фактичне vs Прогнозоване (інтерактивно)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Фактичне'))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines', name='Прогнозоване'))
    fig.update_layout(title=title, xaxis_title='Дата', yaxis_title=f'Курс ({BASE}/{QUOTE})')
    fig.show()

def forecast_future_rf(model, df_feat, forecast_days):
    last_row = df_feat.iloc[-1:].copy()
    preds = []

    for i in range(forecast_days):
        X = last_row.drop(columns=['rate'])
        pred = model.predict(X)[0]
        preds.append(pred)

        new_row = last_row.copy()
        new_row['rate'] = pred
        for lag in [1,2,3,7,14,30]:
            if f'lag_{lag}' in new_row.columns:
                new_row[f'lag_{lag}'] = last_row['rate'].shift(lag) if lag <= len(last_row) else last_row['rate'].values[0]
        new_row['rolling_7'] = df_feat['rate'].iloc[-7:].mean()
        new_row['rolling_30'] = df_feat['rate'].iloc[-30:].mean()
        new_date = df_feat.index[-1] + pd.Timedelta(days=i+1)
        new_row['dayofweek'] = new_date.dayofweek
        new_row['month'] = new_date.month

        last_row = new_row

    future_index = pd.date_range(start=df_feat.index[-1]+pd.Timedelta(days=1), periods=forecast_days)
    return pd.Series(preds, index=future_index)


def main():
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    df_ts, df_feat = build_dataset()

    print("Проводиться розклад на тренд/сезонність...")
    plot_decomposition(df_ts['rate'], model='additive', period=365)

    train_end_date = input(f"Введіть дату, до якої тренувати модель (YYYY-MM-DD, max {df_ts.index[-1].date()}): ")
    try:
        pd.to_datetime(train_end_date)
    except:
        print("Невірний формат дати. Використано останні 90 днів як тест.")
        train_end_date = df_ts.index[-91]

    sarimax_res, sarimax_pred, sarimax_test, sar_metrics = train_sarimax(df_ts['rate'], train_end_date)
    plot_actual_pred(sarimax_test, sarimax_pred, title="SARIMAX: Фактичне vs Прогнозоване",
                     mae=sar_metrics[0], rmse=sar_metrics[1])

    rf_model, rf_pred, rf_test, rf_metrics = train_feature_model(df_feat, train_end_date=train_end_date)
    plot_actual_pred(rf_test, rf_pred, title="RandomForest: Фактичне vs Прогнозоване",
                     mae=rf_metrics[0], rmse=rf_metrics[1])

    joblib.dump(sarimax_res, os.path.join(SAVE_MODEL_PATH, "sarimax_model.joblib"))
    joblib.dump(rf_model, os.path.join(SAVE_MODEL_PATH, "rf_model.joblib"))
    print("Моделі збережено у", SAVE_MODEL_PATH)

    forecast_days = int(input("Скільки днів вперед прогнозувати? "))
    rf_future_series = forecast_future_rf(rf_model, df_feat, forecast_days)
    plot_actual_pred(pd.Series([np.nan]*forecast_days, index=rf_future_series.index),
                    rf_future_series, title=f"RandomForest: Прогноз на {forecast_days} днів")

if __name__ == "__main__":
    main()
