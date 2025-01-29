# Classification
#### Checking and removal data disbalance
```python
from imblearn.under_sampling import RandomUnderSampler


# Check data value counts
new_data = new_data.drop('cluster_id', axis=1)
new_data['group_id'].value_counts()

# Using under_sampling tp removal disbalance (if there is)
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)
```

#### Split data into train and test partitions
```python
from sklearn.model_selection import train_test_split


X = df.drop('group_id', axis=1)
y = df['group_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```


#### Xboost model
```python
# Check metrics
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def xgb_clf(Train_df, Target_df, test_df, true_pred):
    bst = XGBClassifier(n_estimators=20, max_depth=20, learning_rate=1, objective='binary:logistic')
    bst.fit(Train_df, Target_df)
    y_pred = bst.predict(test_df)
    acc = accuracy_score(true_pred, y_pred)
    print(f'Accurancy of xgboost_model: {acc}')
    return bst, y_pred
```

#### RandomForest model
```python
from sklearn.ensemble import RandomForestClassifier


def random_forest_clf(Train_df, Target_df, test_df, true_pred):
    clf = RandomForestClassifier(n_estimators=13, max_depth=5, random_state=42)
    clf.fit(Train_df, Target_df)
    y_pred = clf.predict(test_df)
    acc = accuracy_score(true_pred, y_pred)
    return clf, y_pred
```

#### Logistic Regression model
```python
from sklearn.linear_model import LogisticRegression


def lod_regres_clf(Train_df, Target_df, test_df, true_pred):
    clf = LogisticRegression(random_state=42)
    clf.fit(Train_df, Target_df)
    y_pred = clf.predict(test_df)
    acc = accuracy_score(true_pred, y_pred)
    print(f'Accurancy of log_regres_model: {acc}')
    return clf, y_pred
```

#### Save model
```python
import pickle

def save_model(model):
    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

save_model(rnd_model)
```

#### Advanced training
```python
def advansed_training(conn):
    # Load data from database
    new_data = pd.read_sql('Select * from data', con=conn)
    new_data = new_data.drop('index', axis=1)  # Удаление ненужного столбца
    new_data['group_id'] = new_data['group_id'].astype('int32')
    new_data['date'] = pd.to_datetime(new_data['date']).astype('int64')
    
    # Load previous model
    with open('model.pkl', 'rb') as f:
        tmp_model = pickle.load(f)

    # Split new Data into X and y partitions
    nX = new_data.drop('group_id', axis=1)
    ny = new_data['group_id']

    # Split new Data into train and test partitions
    nX_train, nX_test, ny_train, ny_test = train_test_split(nX, ny, test_size=0.33, random_state=42)

    # Previos model's prediction
    y_tmp_pred = tmp_model.predict(nX_test)

    # Train model
    new_model, new_pred = random_forest_clf(nX_train, ny_train, nX_test, ny_test)

    # Сomparing accuracy metrics between two model.
    if accuracy_score(ny_test, new_pred) > accuracy_score(ny_test, y_tmp_pred):
        print('Model became better')
        with open('model.pkl', 'wb') as f:
            pickle.dump(new_model, f)
    else:
        print('Saving makes no sense')
```


# Regression

#### data preprocessing
```python
warnings.filterwarnings("ignore")

monthly_df = df.copy()
monthly_df['date'] = pd.to_datetime(monthly_df['date'])
monthly_df = monthly_df.set_index('date')
monthly_df = monthly_df.groupby([pd.Grouper(freq="D"), 'station_id'])['num_val'].mean().reset_index()
monthly_df = monthly_df.set_index('date')
monthly_df_arima_df = monthly_df[['num_val']].copy()
monthly_df_arima_df
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
monthly_df_arima_df[['num_val']] = scaler.fit_transform(monthly_df_arima_df[['num_val']])
```

#### Time series visualization
```python
plt.figure(figsize=(15,8))
plt.plot(monthly_df_arima_df['num_val'])
plt.title('Time series Graph')
plt.xlabel('Date')
plt.ylabel('Mean passenger value')
plt.show()
```

#### Data stability check
```python
from statsmodels.tsa.stattools import adfuller

print('Результат теста:')
df_result = adfuller(monthly_df_arima_df['num_val'])
df_labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
for result_value, label in zip(df_result, df_labels):
    print(label + ' : ' + str(result_value))

if df_result[1] <= 0.05:
    print("True.")
else:
    print("False.")
```

#### Check optimal SARIMA parametrs
```python
import itertools

def search_optimal_sarima(time_series):
    order_vals = range(0, 2)  # p
    diff_vals = range(0, 2)   # d
    ma_vals = range(0, 2)     # q

    pdq_combinations = list(itertools.product(order_vals, diff_vals, ma_vals))

    smallest_aic = float("inf")
    optimal_order_param = None

    for order_param in pdq_combinations:
        try:
            sarima_model = sm.tsa.statespace.SARIMAX(
                time_series,
                order=order_param,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model_results = sarima_model.fit(disp=False)

            if model_results.aic < smallest_aic:
                smallest_aic = model_results.aic
                optimal_order_param = order_param
        except Exception as e:
            continue
    print(smallest_aic)
    return optimal_order_param

optimal_order = search_optimal_sarima(monthly_df_arima_df)
print("Оптимальные параметры order:", optimal_order)
```

#### Train optimal model
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(monthly_df_arima_df, order=(0, 0, 0), seasonal_order=(0, 1, 1, 12))
results = model.fit()

print(results.summary())
```

#### Visualization of the model diagnostic diagram
```python
results.plot_diagnostics(figsize=(12, 8))
plt.show()
```

#### Check model MSE error value
```python
st_pred = results.get_prediction(start=pd.to_datetime('2023-01-01'), dynamic=False) 
forecast_values = st_pred.predicted_mean

actual_values = monthly_df_arima_df[:]['num_val']
forecast_mse = ((forecast_values - actual_values) ** 2).mean()
print('Среднеквадратичная ошибка прогноза составляет {}'.format(round(forecast_mse, 2)))
```

#### Predict visualization
```python
plt.figure(figsize=(15,8))
plt.plot(actual_values.index, actual_values, label='Реальные значения', color='blue')
plt.plot(forecast_values.index, forecast_values, label='Спрогнозированные значения', color='red')
plt.title('Реальные и cпрогнозированные значения')
plt.xlabel('Дата')
plt.ylabel('Пассажиры')
plt.legend()
plt.show()
```

#### Prediction of the average station load for the next two years
```python
start_date = pd.to_datetime("2023-09-01")
end_date = start_date + pd.DateOffset(years=2)

steps = (end_date - start_date).days

pred_future = results.get_forecast(steps=steps)
forecast_index = pd.date_range(start=start_date + pd.DateOffset(days=1), periods=steps, freq="D")

forecast_series = pd.Series(pred_future.predicted_mean.values, index=forecast_index)

fig = plt.figure()
plt.plot(forecast_series, label='Mean predict values')
plt.fill_between(forecast_index,
                 pred_future.conf_int().iloc[:, 0],
                 pred_future.conf_int().iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.show()
```

#### Get confidence intervals in original scale 
```python
conf_int_scaled = full_forecast.conf_int()
conf_int_original = scaler.inverse_transform(conf_int_scaled.values)
lower_ci = conf_int_original[:, 0]
upper_ci = conf_int_original[:, 1]

monthly_df_arima_df_pred = monthly_df_arima_df.copy()
orig = scaler.inverse_transform(monthly_df_arima_df['num_val'].values.reshape(-1, 1)).flatten()
monthly_df_arima_df_pred['num_val'] = orig
```

```python
# Visualization
forecast_series_original = scaler.inverse_transform(forecast_series.values.reshape(-1, 1)).flatten()
forecast_series = pd.Series(forecast_series_original, index=forecast_index)

plt.figure(figsize=(10,5))
plt.plot(monthly_df_arima_df_pred.index, monthly_df_arima_df_pred['num_val'], label='Historic data')
plt.plot(forecast_index, forecast_series, label='Predict', color='red')
plt.legend()
plt.show()
```

#### Advanced predict
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def predict_and_inverse_transform(specific_date, fitted_model, scaler):
    last_date = monthly_df_arima_df['num_val'].index[-1]
    steps = (specific_date - last_date).days

    if steps <= 0:
        raise ValueError("The specific date must be after the last date in the time series.")

    pred_future = fitted_model.get_forecast(steps=steps)
    forecast_value_scaled = pred_future.predicted_mean.iloc[-1]

    forecast_value_scaled = np.array(forecast_value_scaled).reshape(-1, 1)
    forecast_value_original = scaler.inverse_transform(forecast_value_scaled)[0][0]

    return forecast_value_original
```

```python
# Predict
specific_date = pd.to_datetime('2024-01-01')
forecast_value = predict_and_inverse_transform(specific_date, results, scaler)
print(f'Прогноз для {specific_date}: {forecast_value}')
```

#### Two week prediction
```python
def two_week_predicts(start_date, results):
    start_date = pd.to_datetime(start_date)
    full_forecast = results.get_forecast(steps=14)
    forecast_index = pd.date_range(start=start_date, periods=additional_steps, freq="D")
    forecast_series_scaled = full_forecast.predicted_mean
    forecast_series_original = scaler.inverse_transform(forecast_series_scaled.values.reshape(-1, 1)).flatten()
    forecast_series = pd.Series(forecast_series_original, index=forecast_index)
    return forecast_series
```

```python
# Visualization
plt.figure(figsize=(10,5))
plt.plot(new.index, new, label='Прогноз', color='red')
plt.legend()
plt.show()
```
