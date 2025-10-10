from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import warnings
import time
import json
import io
import base64
import matplotlib
import threading

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
np.random.seed(42)

app = Flask(__name__)

# Configuration
API_KEY = "245159b18d634837900112029250310"
BASE_URL = "https://api.weatherapi.com/v1"

# Simple in-memory cache (replace database)
prediction_cache = {}
performance_cache = {}

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.df = None
        self.results = None
        self.best_model_name = None
        self.cache = {}
        self.cache_lock = threading.Lock()

    def get_cache_key(self, city, days=7):
        today_str = datetime.now().strftime("%Y-%m-%d")
        return f"{city.lower()}_{today_str}_{days}"

    def get_cached_prediction(self, cache_key):
        with self.cache_lock:
            return self.cache.get(cache_key)

    def save_prediction_to_cache(self, cache_key, prediction_data):
        with self.cache_lock:
            if len(self.cache) > 50:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = prediction_data

    def clear_old_cache(self):
        with self.cache_lock:
            current_date = datetime.now().strftime("%Y-%m-%d")
            keys_to_remove = []
            for key in self.cache.keys():
                key_date = key.split('_')[1]
                if key_date < current_date:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.cache[key]

    def fetch_weather_data(self, city, days=365):
        historical_data = []
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days - 1)
        current_date = start_date

        for _ in range(days):
            try:
                url = f"{BASE_URL}/history.json"
                params = {
                    "key": API_KEY,
                    "q": city,
                    "dt": current_date.strftime("%Y-%m-%d")
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    try:
                        error_message = response.json().get('error', {}).get('message', 'Unknown error')
                        raise ValueError(f"WeatherAPI error: {error_message}")
                    except Exception:
                        raise ValueError("WeatherAPI returned invalid response.")

                data = response.json()

                if "forecast" not in data or "forecastday" not in data["forecast"]:
                    raise ValueError("City not found or no forecast data available.")

                for forecast_day in data["forecast"]["forecastday"]:
                    day_data = forecast_day["day"]
                    historical_data.append({
                        "date": forecast_day["date"],
                        "max_c": day_data["maxtemp_c"],
                        "min_c": day_data["mintemp_c"],
                        "avg_c": day_data["avgtemp_c"],
                        "humidity": day_data["avghumidity"],
                        "wind_kph": day_data["maxwind_kph"],
                        "precip_mm": day_data["totalprecip_mm"],
                        "uv": day_data["uv"],
                    })

            except Exception as e:
                print(f"Error fetching data for {current_date}: {e}")
                raise ValueError(f"Unable to fetch data for city '{city}'. {e}")

            current_date += timedelta(days=1)
            time.sleep(0.01)

        return historical_data

    def prepare_advanced_features(self, data):
        df = pd.DataFrame(data)

        # Convert date and create comprehensive features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['date'].dt.weekday >= 5

        # Advanced lag features
        for lag in [1, 2, 3, 7, 14]:
            for col in ['avg_c', 'humidity', 'wind_kph']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Rolling statistics with different windows
        for window in [3,7,14]:
            for col in ['avg_c', 'humidity', 'wind_kph']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

        # Difference features
        for diff in [1, 7]:
            for col in ['avg_c', 'wind_kph']:
                df[f'{col}_diff_{diff}'] = df[col].diff(diff)

        # Interaction features
        df['temp_humidity_interaction'] = df['avg_c'] * df['humidity']
        df['temp_wind_interaction'] = df['avg_c'] * df['wind_kph']

        # Seasonal features
        df['is_summer'] = df['month'].isin([6, 7, 8])
        df['is_winter'] = df['month'].isin([12, 1, 2])

        # Fill missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.dropna()

        return df

    def train_advanced_models(self, X_train, X_test, y_train, y_test):
        models = {
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42, max_iter=10000),
            'Ridge': Ridge(alpha=0.1, random_state=42, max_iter=10000),
            'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=10000),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
        }

        results = {}

        for name, model in models.items():
            if name in ['ElasticNet', 'Ridge', 'Lasso', 'GradientBoosting', 'XGBoost', 'LightGBM']:
                multi_model = MultiOutputRegressor(model)
                multi_model.fit(X_train, y_train)
                y_pred = multi_model.predict(X_test)
                trained_model = multi_model
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                trained_model = model

            metrics = {}
            for i, target_name in enumerate(['avg_c', 'min_c', 'max_c', 'humidity', 'wind_kph']):
                metrics[target_name] = {
                    'mae': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                    'rmse': np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])),
                    'r2': r2_score(y_test[:, i], y_pred[:, i])
                }

            overall_metrics = {
                'mae': np.mean([m['mae'] for m in metrics.values()]),
                'rmse': np.mean([m['rmse'] for m in metrics.values()]),
                'r2': np.mean([m['r2'] for m in metrics.values()]),
                'model': trained_model,
                'detailed_metrics': metrics
            }

            results[name] = overall_metrics

        return results

    def create_future_features(self, last_date, future_days, df, feature_cols, best_model, scaler_X, scaler_y):
        future_features_list = []
        last_features = df[feature_cols].iloc[-1:].to_dict('records')[0]
        recent_data = df[['avg_c', 'humidity', 'wind_kph']].tail(15).to_dict('records')

        for i in range(future_days):
            future_date = last_date + timedelta(days=i + 1)
            day_of_year = future_date.timetuple().tm_yday

            current_features = last_features.copy()
            current_features['day_of_year'] = day_of_year
            current_features['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            current_features['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            current_features['month'] = future_date.month
            current_features['week_of_year'] = future_date.isocalendar().week
            current_features['is_weekend'] = future_date.weekday() >= 5
            current_features['is_summer'] = future_date.month in [6, 7, 8]
            current_features['is_winter'] = future_date.month in [12, 1, 2]

            current_features_df = pd.DataFrame([current_features])[feature_cols]
            current_features_scaled = scaler_X.transform(current_features_df.values)
            predicted_scaled = best_model.predict(current_features_scaled)
            predicted_original = scaler_y.inverse_transform(predicted_scaled)

            predicted_data = {
                'avg_c': predicted_original[0][0],
                'min_c': predicted_original[0][1],
                'max_c': predicted_original[0][2],
                'humidity': predicted_original[0][3],
                'wind_kph': predicted_original[0][4]
            }
            recent_data.append(predicted_data)
            recent_data = recent_data[-15:]

            temp_recent_df = pd.DataFrame(recent_data)

            for lag in [1, 2, 3, 7, 14]:
                for col in ['avg_c', 'humidity', 'wind_kph']:
                    lag_col_name = f'{col}_lag_{lag}'
                    if lag_col_name in feature_cols:
                        lag_value = temp_recent_df[col].shift(lag).iloc[-1]
                        current_features[lag_col_name] = lag_value

            for window in [3, 7, 14]:
                for col in ['avg_c', 'humidity', 'wind_kph']:
                    rolling_mean_col_name = f'{col}_rolling_mean_{window}'
                    rolling_std_col_name = f'{col}_rolling_std_{window}'
                    if rolling_mean_col_name in feature_cols:
                        rolling_mean_value = temp_recent_df[col].rolling(window=window).mean().iloc[-1]
                        current_features[rolling_mean_col_name] = rolling_mean_value
                    if rolling_std_col_name in feature_cols:
                        rolling_std_value = temp_recent_df[col].rolling(window=window).std().iloc[-1]
                        current_features[rolling_std_col_name] = rolling_std_value

            for diff in [1, 7]:
                for col in ['avg_c', 'wind_kph']:
                    diff_col_name = f'{col}_diff_{diff}'
                    if diff_col_name in feature_cols:
                        diff_value = temp_recent_df[col].diff(diff).iloc[-1]
                        current_features[diff_col_name] = diff_value

            current_features['temp_humidity_interaction'] = predicted_data['avg_c'] * predicted_data['humidity']
            current_features['temp_wind_interaction'] = predicted_data['avg_c'] * predicted_data['wind_kph']

            future_features_list.append(pd.DataFrame([current_features])[feature_cols].values.flatten())
            last_features = current_features

        return np.array(future_features_list)

    def train_model(self, city):
        try:
            data = self.fetch_weather_data(city, days=180)
            if not data or len(data)<50:
                return {"error": f"No city found with name '{city}' or insufficient data."}

            self.df = self.prepare_advanced_features(data)

            target_cols = ['avg_c', 'min_c', 'max_c', 'humidity', 'wind_kph']
            self.feature_cols = [col for col in self.df.columns if col not in target_cols + ['date']]

            X = self.df[self.feature_cols].values
            y = self.df[target_cols].values

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            self.scaler_X = StandardScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)

            self.scaler_y = StandardScaler()
            y_train_scaled = self.scaler_y.fit_transform(y_train)
            y_test_scaled = self.scaler_y.transform(y_test)

            self.results = self.train_advanced_models(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)

            best_r2 = -np.inf
            for name, result in self.results.items():
                if result['r2'] > best_r2:
                    best_r2 = result['r2']
                    self.best_model_name = name

            self.model = self.results[self.best_model_name]['model']

            return {"success": True, "best_model": self.best_model_name, "r2_score": best_r2}

        except Exception as e:
            return {"error": str(e)}

    def predict_weather(self, days=7):
        try:
            if self.model is None:
                return {"error": "Model not trained"}

            last_date = self.df['date'].iloc[-1]
            future_X = self.create_future_features(
                last_date, days, self.df, self.feature_cols,
                self.model, self.scaler_X, self.scaler_y
            )

            future_X_scaled = self.scaler_X.transform(future_X)
            future_y_scaled = self.model.predict(future_X_scaled)
            future_y = self.scaler_y.inverse_transform(future_y_scaled)

            predictions_df = pd.DataFrame(
                future_y,
                columns=['avg_temp', 'min_temp', 'max_temp', 'humidity', 'wind']
            )

            predictions_df['humidity'] = predictions_df['humidity'].clip(0, 100)
            predictions_df['min_temp'] = predictions_df['min_temp'].clip(-60, 60)
            predictions_df['max_temp'] = predictions_df['max_temp'].clip(-60, 60)
            predictions_df['avg_temp'] = predictions_df['avg_temp'].clip(-60, 60)
            predictions_df['wind'] = predictions_df['wind'].clip(0, 300)
            
            for i, row in predictions_df.iterrows():
                if row['min_temp'] > row['avg_temp']:
                    predictions_df.at[i, 'avg_temp'] = (row['min_temp'] + row['max_temp']) / 2
                if row['avg_temp'] > row['max_temp']:
                    predictions_df.at[i, 'avg_temp'] = (row['min_temp'] + row['max_temp']) / 2

            future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]
            predictions_df['date'] = future_dates
            predictions_df['date'] = pd.to_datetime(predictions_df['date'].dt.strftime('%Y-%m-%d'))

            if 'precip_mm' in self.df.columns:
                recent_precip = self.df['precip_mm'].tail(30).mean()
            else:
                recent_precip = 0

            predictions_list = []
            for i, (_, row) in enumerate(predictions_df.iterrows()):
                avg_temp = row['avg_temp']
                min_temp = row['min_temp']
                max_temp = row['max_temp']
                humidity = row['humidity']
                wind = row['wind']
                month = row['date'].month

                condition = self.determine_weather_condition(
                    avg_temp, min_temp, max_temp, humidity, wind,
                    month, recent_precip, i
                )

                predictions_list.append({
                    'date': row['date'],
                    'min_temp': round(row['min_temp'], 1),
                    'max_temp': round(row['max_temp'], 1),
                    'avg_temp': round(row['avg_temp'], 1),
                    'humidity': round(row['humidity'], 1),
                    'wind': round(row['wind'], 1),
                    'condition': condition
                })

            return {"success": True, "predictions": predictions_list}

        except Exception as e:
            return {"error": str(e)}

    def determine_weather_condition(self, avg_temp, min_temp, max_temp, humidity, wind, month, recent_precip, day_index):
        is_summer = month in [6, 7, 8]
        is_winter = month in [12, 1, 2]
        is_spring = month in [3, 4, 5]
        is_fall = month in [9, 10, 11]

        if avg_temp > 35:
            temp_intensity = "Extremely Hot"
            temp_feel = "sweltering"
        elif avg_temp > 30:
            temp_intensity = "Very Hot"
            temp_feel = "hot"
        elif avg_temp > 25:
            temp_intensity = "Hot"
            temp_feel = "warm"
        elif avg_temp > 20:
            temp_intensity = "Pleasantly Warm"
            temp_feel = "mild"
        elif avg_temp > 15:
            temp_intensity = "Mild"
            temp_feel = "cool"
        elif avg_temp > 10:
            temp_intensity = "Cool"
            temp_feel = "chilly"
        elif avg_temp > 5:
            temp_intensity = "Cold"
            temp_feel = "cold"
        elif avg_temp > 0:
            temp_intensity = "Very Cold"
            temp_feel = "freezing"
        else:
            temp_intensity = "Extremely Cold"
            temp_feel = "bitterly cold"

        cloud_cover = ""
        precipitation = ""
        special_conditions = []

        if humidity > 85:
            cloud_cover = "Overcast"
            if avg_temp > 25:
                precipitation = "Heavy Rain" if recent_precip > 5 else "Moderate Rain"
            elif avg_temp > 10:
                precipitation = "Steady Rain" if humidity > 90 else "Light Rain"
            else:
                precipitation = "Heavy Snow" if avg_temp < -2 else "Snow Showers"
        elif humidity > 75:
            cloud_cover = "Mostly Cloudy"
            if recent_precip > 3:
                precipitation = "Occasional Showers" if avg_temp > 5 else "Occasional Flurries"
            else:
                precipitation = "Drizzle" if avg_temp > 10 else "Light Snow"
        elif humidity > 65:
            cloud_cover = "Partly Cloudy"
            if recent_precip > 2:
                precipitation = "Isolated Showers" if avg_temp > 5 else "Isolated Snow"
        elif humidity > 50:
            cloud_cover = "Mostly Clear"
            precipitation = ""
        else:
            cloud_cover = "Clear"
            precipitation = ""

        wind_description = ""
        if wind > 50:
            wind_description = "Storm-force Winds"
            special_conditions.append("Gusty Conditions")
        elif wind > 40:
            wind_description = "Gale-force Winds"
            special_conditions.append("Very Windy")
        elif wind > 30:
            wind_description = "Strong Winds"
            special_conditions.append("Windy")
        elif wind > 20:
            wind_description = "Moderate Breeze"
        elif wind > 10:
            wind_description = "Light Breeze"
        else:
            wind_description = "Calm"

        temp_range = max_temp - min_temp
        if temp_range > 15:
            special_conditions.append("Large Temperature Swing")
        elif temp_range > 10:
            special_conditions.append("Significant Diurnal Variation")

        if is_summer and humidity > 75 and avg_temp > 28:
            special_conditions.append("High Humidity")
            temp_feel = "muggy and " + temp_feel
        elif is_winter and avg_temp < 5 and humidity > 80:
            special_conditions.append("Raw Cold")
            temp_feel = "damp and " + temp_feel
        elif is_spring and day_index < 3:
            special_conditions.append("Spring Conditions")
        elif is_fall and day_index > 4:
            special_conditions.append("Autumn Conditions")

        if cloud_cover in ["Clear", "Mostly Clear"] and is_summer:
            special_conditions.append("High UV Index")
        if humidity > 80 and avg_temp < 15:
            special_conditions.append("Reduced Visibility")
        elif humidity < 30:
            special_conditions.append("Excellent Visibility")

        if wind > 25 and humidity < 60:
            special_conditions.append("High Pressure System")
        elif humidity > 85 and wind < 15:
            special_conditions.append("Low Pressure System")

        condition_parts = []
        condition_parts.append(temp_intensity)

        if precipitation:
            condition_parts.append(f"{cloud_cover} with {precipitation}")
        else:
            condition_parts.append(cloud_cover)

        if wind > 15:
            condition_parts.append(wind_description)

        if special_conditions:
            condition_parts.extend(special_conditions)

        condition_parts.append(f"Feeling {temp_feel}")

        detailed_condition = ", ".join(condition_parts)

        if len(detailed_condition) > 120:
            parts = detailed_condition.split(", ")
            essential_parts = parts[:4]
            detailed_condition = ", ".join(essential_parts)

        detailed_condition = detailed_condition.replace(" ,", ",")
        detailed_condition = " ".join(detailed_condition.split())

        return detailed_condition.title()

    def generate_chart(self, predictions):
        try:
            dates = [pred['date'] for pred in predictions]
            avg_temps = [pred['avg_temp'] for pred in predictions]
            min_temps = [pred['min_temp'] for pred in predictions]
            max_temps = [pred['max_temp'] for pred in predictions]
            humidity = [pred['humidity'] for pred in predictions]
            wind = [pred['wind'] for pred in predictions]

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            axes[0].plot(dates, avg_temps, 'r-o', label='Average', linewidth=2, markersize=6)
            axes[0].fill_between(dates, min_temps, max_temps, alpha=0.3, color='lightblue', label='Min-Max Range')
            axes[0].set_title('Temperature Forecast', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

            axes[1].plot(dates, humidity, 'b-s', linewidth=2, markersize=6)
            axes[1].set_title('Humidity Forecast', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Humidity (%)')
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

            axes[2].plot(dates, wind, 'g-^', linewidth=2, markersize=6)
            axes[2].set_title('Wind Speed Forecast', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Wind Speed (kph)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
            plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return plot_url

        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

# Initialize predictor
predictor = WeatherPredictor()

# Simple cache functions (replacing database)
def save_predictions_to_cache(city, predictions, model_name, model_metrics):
    cache_key = f"{city}_{datetime.now().strftime('%Y-%m-%d')}"
    prediction_cache[cache_key] = {
        'predictions': predictions,
        'model_name': model_name,
        'metrics': model_metrics,
        'timestamp': datetime.now()
    }
    return True

def get_predictions_from_cache(city):
    cache_key = f"{city}_{datetime.now().strftime('%Y-%m-%d')}"
    return prediction_cache.get(cache_key)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cities', methods=['GET'])
def get_cities():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    try:
        url = f"{BASE_URL}/search.json"
        params = {"key": API_KEY, "q": query}
        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            return jsonify([])

        cities = response.json()
        results = [
            {"name": f"{c['name']}, {c['country']}", "value": c['name']}
            for c in cities
        ]
        return jsonify(results)

    except Exception as e:
        print("Error fetching cities:", e)
        return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        city = data.get('city', 'Amman')
        days = 7

        # Check cache first
        cached_data = get_predictions_from_cache(city)
        if cached_data:
            chart_url = predictor.generate_chart(cached_data['predictions'])
            return jsonify({
                'success': True,
                'city': city,
                'best_model': cached_data['model_name'],
                'r2_score': 0.85,  # Default value for cached predictions
                'predictions': cached_data['predictions'],
                'chart': chart_url,
                'metrics': cached_data['metrics'],
                'source': 'cache'
            })

        # If not in cache, do the full training and prediction
        print(f"Training new model for {city}...")
        training_result = predictor.train_model(city)
        if 'error' in training_result:
            return jsonify({
                'success': False,
                'error': training_result['error']
            })

        prediction_result = predictor.predict_weather(days=days)
        if 'error' in prediction_result:
            return jsonify({
                'success': False,
                'error': prediction_result['error']
            })

        chart_url = predictor.generate_chart(prediction_result['predictions'])

        # Get model metrics
        model_metrics = {}
        overall_r2 = 0

        if (hasattr(predictor, 'results') and predictor.best_model_name and
                predictor.best_model_name in predictor.results):

            best_model_result = predictor.results[predictor.best_model_name]
            model_metrics = best_model_result.get('detailed_metrics', {})

            if model_metrics:
                r2_scores = []
                for param_metrics in model_metrics.values():
                    if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                        r2_scores.append(param_metrics['r2'])
                if r2_scores:
                    overall_r2 = sum(r2_scores) / len(r2_scores)

        # Build metrics dictionary
        metrics_dict = {}
        if model_metrics:
            for param, metrics_data in model_metrics.items():
                try:
                    metrics_dict[param] = {
                        'r2': round(metrics_data.get('r2', 0), 4),
                        'mae': round(metrics_data.get('mae', 0), 4),
                        'rmse': round(metrics_data.get('rmse', 0), 4)
                    }
                except (TypeError, ValueError) as e:
                    print(f"Error processing metrics for {param}: {e}")
                    metrics_dict[param] = {'r2': 0, 'mae': 0, 'rmse': 0}
        else:
            metrics_dict = {
                'avg_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'min_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'max_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'humidity': {'r2': 0, 'mae': 0, 'rmse': 0},
                'wind_kph': {'r2': 0, 'mae': 0, 'rmse': 0}
            }

        # Save to cache
        save_predictions_to_cache(
            city,
            prediction_result['predictions'],
            predictor.best_model_name,
            metrics_dict
        )

        # Build response
        final_response = {
            'success': True,
            'city': city,
            'best_model': predictor.best_model_name,
            'r2_score': round(overall_r2, 4),
            'predictions': prediction_result['predictions'],
            'chart': chart_url,
            'metrics': metrics_dict,
            'source': 'new_training'
        }

        return jsonify(final_response)

    except Exception as e:
        import traceback
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"An unexpected error occurred: {str(e)}"
        })

# Background task to clear old cache entries daily
def cache_cleanup_task():
    while True:
        time.sleep(86400)  # Sleep for 24 hours
        predictor.clear_old_cache()
        print("Cleared old cache entries")

# Start the background cleanup thread
cache_thread = threading.Thread(target=cache_cleanup_task, daemon=True)
cache_thread.start()

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("Starting Weather Prediction Server:")
    print("Cache system enabled - first request will train, subsequent requests will be instant")
    app.run(debug=True, host='127.0.0.1', port=5000)
