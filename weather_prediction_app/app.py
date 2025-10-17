from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import requests
from datetime import timedelta
import numpy as np
import pandas as pd
import sqlite3
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import threading
from pathlib import Path
import shutil

app = Flask(__name__)

# POSTGRESQL DATABASE CONFIGURATION - THIS WILL PERSIST!
def get_database_uri():
    # Priority 1: Supabase
    supabase_url = os.environ.get('SUPABASE_URL')
    if supabase_url and supabase_url != 'postgresql://postgres:6d4FxFGAbX9tntdD@db.cvznilxsqbexhywisntv.supabase.co:5432/postgres':
        # Make sure it's not the template URL
        if '[YOUR-PASSWORD]' not in supabase_url:
            print("✅ Using Supabase PostgreSQL")
            return supabase_url
    
    # Priority 2: Check if we're on Render but no database
    if 'RENDER' in os.environ:
        # Fallback to SQLite on Render
        print("⚠️ No Supabase URL found, using SQLite on Render")
        instance_path = os.path.join(os.getcwd(), 'instance')
        Path(instance_path).mkdir(exist_ok=True)
        db_path = os.path.join(instance_path, 'weather_predictions.db')
        return f"sqlite:///{db_path}"
    
    # Local development
    return "sqlite:///weather_predictions.db"
    
db = SQLAlchemy(app)

# Define models AFTER db initialization
class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    prediction_date = db.Column(db.Date, nullable=False)
    generation_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    model_version = db.Column(db.String(50), nullable=False)
    min_temp = db.Column(db.Float, nullable=False)
    max_temp = db.Column(db.Float, nullable=False)
    avg_temp = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    condition = db.Column(db.String(100), nullable=False)
    is_current = db.Column(db.Boolean, default=True)
    version = db.Column(db.Integer, default=1)

class ModelPerformance(db.Model):
    __tablename__ = "model_performance"
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    model_name = db.Column(db.String(50), nullable=False)
    r2_score = db.Column(db.Float, nullable=False)
    mae = db.Column(db.Float, nullable=False)
    rmse = db.Column(db.Float, nullable=False)
    detailed_metrics = db.Column(db.JSON, nullable=False)

API_KEY = "387452ec05eb4b38b74113541251610"
BASE_URL = "https://api.weatherapi.com/v1"

# Keep-alive system
class KeepAliveManager:
    def __init__(self):
        self.enabled = 'RENDER' in os.environ
        self.app_url = "https://skysense-20a1.onrender.com"
        
    def ping_self(self):
        """Ping the health endpoint"""
        try:
            response = requests.get(f"{self.app_url}/health", timeout=10)
            print(f"KeepAlive: Ping successful - {response.status_code} at {time.strftime('%H:%M:%S')}")
            return True
        except Exception as e:
            print(f"KeepAlive: Ping failed - {e}")
            return False
    
    def start_keep_alive(self):
        """Start the keep-alive system"""
        if not self.enabled:
            print("KeepAlive: Running locally, no need for keep-alive")
            return
            
        print(f"KeepAlive: Starting for {self.app_url}")
        ping_thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        ping_thread.start()
    
    def _keep_alive_loop(self):
        """Background loop to keep the app alive"""
        while True:
            self.ping_self()
            time.sleep(600)  # 10 minutes

# Create keep-alive manager instance
keep_alive_manager = KeepAliveManager()

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
        """Generate a unique key for the cache based on city and date."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        return f"{city.lower()}_{today_str}_{days}"

    def get_cached_prediction(self, cache_key):
        """Retrieve a prediction from the cache if it exists and is valid."""
        with self.cache_lock:
            return self.cache.get(cache_key)

    def save_prediction_to_cache(self, cache_key, prediction_data):
        """Save a prediction to the cache."""
        with self.cache_lock:
            if len(self.cache) > 50:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = prediction_data

    def clear_old_cache(self):
        """Clear cache entries older than 2 days"""
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
        """Fetch weather data from API with error handling for invalid city"""
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
        """Prepare advanced features for machine learning"""
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['date'].dt.weekday >= 5

        for lag in [1, 2, 3, 7, 14]:
            for col in ['avg_c', 'humidity', 'wind_kph']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        for window in [3,7,14]:
            for col in ['avg_c', 'humidity', 'wind_kph']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

        for diff in [1, 7]:
            for col in ['avg_c', 'wind_kph']:
                df[f'{col}_diff_{diff}'] = df[col].diff(diff)

        df['temp_humidity_interaction'] = df['avg_c'] * df['humidity']
        df['temp_wind_interaction'] = df['avg_c'] * df['wind_kph']
        df['is_summer'] = df['month'].isin([6, 7, 8])
        df['is_winter'] = df['month'].isin([12, 1, 2])

        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.dropna()

        return df

    def train_advanced_models(self, X_train, X_test, y_train, y_test):
        """Train advanced machine learning models with tuned parameters"""
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
            future_date = last_date + timedelta(days=i + 1)  # This is the key line!
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

            # Convert to DataFrame for scaling and prediction
            current_features_df = pd.DataFrame([current_features])[feature_cols]
            current_features_scaled = scaler_X.transform(current_features_df.values)
            predicted_scaled = best_model.predict(current_features_scaled)
            predicted_original = scaler_y.inverse_transform(predicted_scaled)
            # Update the recent_data with the new prediction
            predicted_data = {
            'avg_c': predicted_original[0][0],
            'min_c': predicted_original[0][1],
            'max_c': predicted_original[0][2],
            'humidity': predicted_original[0][3],
            'wind_kph': predicted_original[0][4]
            }
            recent_data.append(predicted_data)
            recent_data = recent_data[-15:]
            # Create a temporary DataFrame from recent_data to calculate updated features
            temp_recent_df = pd.DataFrame(recent_data)

            # Recalculate lag and rolling features
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

            # Recalculate difference features
            for diff in [1, 7]:
                for col in ['avg_c', 'wind_kph']:
                    diff_col_name = f'{col}_diff_{diff}'
                    if diff_col_name in feature_cols:
                        diff_value = temp_recent_df[col].diff(diff).iloc[-1]
                        current_features[diff_col_name] = diff_value

            # Recalculate interaction features
            current_features['temp_humidity_interaction'] = predicted_data['avg_c'] * predicted_data['humidity']
            current_features['temp_wind_interaction'] = predicted_data['avg_c'] * predicted_data['wind_kph']

            # Store the updated features
            future_features_list.append(pd.DataFrame([current_features])[feature_cols].values.flatten())
            last_features = current_features

        return np.array(future_features_list)


        
    def train_model(self, city):
        """Complete model training pipeline"""
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
        """Generate weather predictions starting from TODAY"""
        try:
            if self.model is None:
                return {"error": "Model not trained"}

            # FIX: Start from TODAY instead of last historical date
            today = datetime.now().date()
            print(f"Generating predictions starting from: {today}")
            
            # Use today as the starting point
            future_X = self.create_future_features(
                pd.to_datetime(today),  # CHANGED: Start from today
                days, 
                self.df, 
                self.feature_cols,
                self.model, 
                self.scaler_X, 
                self.scaler_y
            )

            # Scale and predict
            future_X_scaled = self.scaler_X.transform(future_X)
            future_y_scaled = self.model.predict(future_X_scaled)
            future_y = self.scaler_y.inverse_transform(future_y_scaled)

            # Create predictions DataFrame
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

            # FIX: Generate dates starting from TODAY
            future_dates = [today + timedelta(days=i) for i in range(days)]
            predictions_df['date'] = future_dates
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])

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

            print(f"Generated predictions for: {[d.strftime('%Y-%m-%d') for d in future_dates]}")
            return {"success": True, "predictions": predictions_list}

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}

    def determine_weather_condition(self, avg_temp, min_temp, max_temp, humidity, wind, month, recent_precip, day_index):
        """Determine detailed weather condition based on multiple factors"""
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
        """Generate weather prediction chart"""
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

predictor = WeatherPredictor()

def cleanup_old_predictions():
    """Mark old predictions as not current when they're no longer relevant"""
    try:
        today = datetime.now().date()
        print(f"🧹 Cleaning up predictions older than {today}")
        
        outdated = Prediction.query.filter(
            Prediction.prediction_date < today,
            Prediction.is_current == True
        ).update({'is_current': False}, synchronize_session=False)
        
        if outdated:
            print(f"Marked {outdated} outdated predictions as not current")
        else:
            print("No outdated predictions found")
            
        db.session.commit()
    except Exception as e:
        print(f"Error cleaning up old predictions: {e}")

def save_predictions_to_db(city, predictions, model_name, model_metrics):
    """Save predictions to the database, preserving historical data"""
    try:
        generation_time = datetime.now()
        today = datetime.now().date()

        # First, mark all previous predictions for this city as not current
        Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= today,
            Prediction.is_current == True
        ).update({'is_current': False}, synchronize_session=False)

        max_version = db.session.query(db.func.max(Prediction.version)).filter_by(
            city=city
        ).scalar() or 0

        new_version = max_version + 1

        for prediction in predictions:
            if isinstance(prediction['date'], str):
                prediction_date = datetime.strptime(prediction['date'], '%Y-%m-%d').date()
            else:
                prediction_date = prediction['date'].date()

            if prediction_date >= today:
                new_prediction = Prediction(
                    city=city,
                    prediction_date=prediction_date,
                    generation_timestamp=generation_time,
                    model_version=model_name or "Unknown",
                    min_temp=prediction['min_temp'],
                    max_temp=prediction['max_temp'],
                    avg_temp=prediction['avg_temp'],
                    humidity=prediction['humidity'],
                    wind_speed=prediction['wind'],
                    condition=prediction['condition'],
                    is_current=True,
                    version=new_version
                )
                db.session.add(new_prediction)

        overall_r2 = 0
        if model_metrics and isinstance(model_metrics, dict):
            r2_scores = []
            for param_metrics in model_metrics.values():
                if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                    r2_scores.append(param_metrics['r2'])
            if r2_scores:
                overall_r2 = sum(r2_scores) / len(r2_scores)

        overall_mae = 0
        overall_rmse = 0
        if model_metrics and isinstance(model_metrics, dict):
            mae_scores = []
            rmse_scores = []
            for param_metrics in model_metrics.values():
                if isinstance(param_metrics, dict):
                    if 'mae' in param_metrics:
                        mae_scores.append(param_metrics['mae'])
                    if 'rmse' in param_metrics:
                        rmse_scores.append(param_metrics['rmse'])

            if mae_scores:
                overall_mae = sum(mae_scores) / len(mae_scores)
            if rmse_scores:
                overall_rmse = sum(rmse_scores) / len(rmse_scores)

        performance = ModelPerformance(
            city=city,
            timestamp=generation_time,
            model_name=model_name or "Unknown",
            r2_score=round(overall_r2, 4),
            mae=round(overall_mae, 4),
            rmse=round(overall_rmse, 4),
            detailed_metrics=model_metrics
        )
        db.session.add(performance)

        db.session.commit()
        
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving to database: {e}")
        return False

def get_latest_predictions_from_db(city, days=7):
    """Retrieve the latest predictions from the database"""
    try:
        today = datetime.now().date()

        # Get the most recent current predictions for this city starting from today
        predictions = Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= today,
            Prediction.is_current == True
        ).order_by(Prediction.prediction_date.asc()).limit(days).all()

        if not predictions or len(predictions) < days:
            return None

        # Verify we have a complete 7-day forecast starting from today
        prediction_dates = [pred.prediction_date for pred in predictions]
        expected_dates = [today + timedelta(days=i) for i in range(days)]

        if prediction_dates != expected_dates:
            return None

        # Convert to list of dictionaries
        result = []
        for pred in predictions:
            result.append({
                'date': pred.prediction_date,
                'min_temp': pred.min_temp,
                'max_temp': pred.max_temp,
                'avg_temp': pred.avg_temp,
                'humidity': pred.humidity,
                'wind': pred.wind_speed,
                'condition': pred.condition
            })

        return result
    except Exception as e:
        print(f"Error retrieving from database: {e}")
        return None
        
#Initialize predictor
predictor = WeatherPredictor()

@app.route('/health')
def health_check():
    """Minimal health check for uptime monitoring"""
    return 'OK', 200

@app.route('/debug-current')
def debug_current():
    """Debug route to see what's currently in the database"""
    try:
        today = datetime.now().date()
        city = "Amman"
        
        print(f"DEBUG: Today is {today}")
        
        all_current = Prediction.query.filter(
            Prediction.is_current == True
        ).all()
        
        current_data = []
        for pred in all_current:
            current_data.append({
                'city': pred.city,
                'date': pred.prediction_date.isoformat(),
                'generated': pred.generation_timestamp.date().isoformat(),
                'is_current': pred.is_current
            })
        
        return jsonify({
            'today': today.isoformat(),
            'database_type': 'PostgreSQL' if 'postgresql' in app.config["SQLALCHEMY_DATABASE_URI"] else 'SQLite',
            'all_current_predictions': current_data,
            'total_current': len(current_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/force-reset-today')
def force_reset_today():
    """Temporary route to force new predictions for today"""
    try:
        today = datetime.now().date()
        city = "Amman"
        
        reset_count = Prediction.query.filter(
            Prediction.is_current == True
        ).update({'is_current': False}, synchronize_session=False)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Reset {reset_count} predictions. Next prediction will generate fresh data for {today}',
            'today': today.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        city = data.get('city', 'Amman')
        days = 7

        # Check database first
        db_predictions = get_latest_predictions_from_db(city, days)
        if db_predictions:
            # Initialize default values
            metrics = {}
            model_name = "Unknown"
            r2_score = 0

            # Try to get performance metrics
            latest_performance = ModelPerformance.query.filter_by(
                city=city
            ).order_by(ModelPerformance.timestamp.desc()).first()

            if latest_performance:
                # Handle both old and new metric formats
                metrics = latest_performance.detailed_metrics or {}
                model_name = latest_performance.model_name or "Unknown"
                r2_score = latest_performance.r2_score or 0

                # If R² is 0 but we have detailed metrics, calculate it
                if r2_score == 0 and metrics:
                    r2_scores = []
                    for param_metrics in metrics.values():
                        if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                            r2_scores.append(param_metrics['r2'])
                    if r2_scores:
                        r2_score = sum(r2_scores) / len(r2_scores)

            # Generate chart for the cached predictions
            chart_url = predictor.generate_chart(db_predictions)

            return jsonify({
                'success': True,
                'city': city,
                'best_model': model_name,
                'r2_score': round(r2_score, 4),
                'predictions': db_predictions,
                'chart': chart_url,
                'metrics': metrics,
                'source': 'database'
            })

        # If not in database, do the full training and prediction
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

        # Safely get model metrics
        model_metrics = {}
        overall_r2 = 0

        if (hasattr(predictor, 'results') and predictor.best_model_name and
                predictor.best_model_name in predictor.results):

            best_model_result = predictor.results[predictor.best_model_name]
            model_metrics = best_model_result.get('detailed_metrics', {})

            # Calculate overall R² score
            if model_metrics:
                r2_scores = []
                for param_metrics in model_metrics.values():
                    if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                        r2_scores.append(param_metrics['r2'])
                if r2_scores:
                    overall_r2 = sum(r2_scores) / len(r2_scores)

        # Build metrics dictionary with proper structure
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
            print("Warning: No detailed metrics available")
            # Create default metrics structure
            metrics_dict = {
                'avg_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'min_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'max_c': {'r2': 0, 'mae': 0, 'rmse': 0},
                'humidity': {'r2': 0, 'mae': 0, 'rmse': 0},
                'wind_kph': {'r2': 0, 'mae': 0, 'rmse': 0}
            }

        # Save to database
        save_success = save_predictions_to_db(
            city,
            prediction_result['predictions'],
            predictor.best_model_name,
            metrics_dict
        )

        if not save_success:
            print("Warning: Failed to save predictions to database")

        # Build the final response object
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


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    try:
        predictor.cache.clear()
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    try:
        stats = {
            'cache_size': len(predictor.cache),
            'cached_cities': list(set(key.split('_')[0] for key in predictor.cache.keys())),
            'total_entries': len(predictor.cache)
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def cache_cleanup_task():
    while True:
        time.sleep(86400)
        predictor.clear_old_cache()
        print("Cleared old cache entries")

cache_thread = threading.Thread(target=cache_cleanup_task, daemon=True)
cache_thread.start()


@app.route('/history/<city>', methods=['GET'])
def get_history(city):
    """Get historical predictions for analysis"""
    try:
        # Get the last 30 days of predictions
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        predictions = Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= start_date,
            Prediction.prediction_date <= end_date
        ).order_by(Prediction.prediction_date.asc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found'
            })

        # Format the response
        result = []
        for pred in predictions:
            result.append({
                'prediction_date': pred.prediction_date.isoformat(),
                'generation_timestamp': pred.generation_timestamp.isoformat(),
                'model_version': pred.model_version,
                'min_temp': pred.min_temp,
                'max_temp': pred.max_temp,
                'avg_temp': pred.avg_temp,
                'humidity': pred.humidity,
                'wind_speed': pred.wind_speed,
                'condition': pred.condition
            })

        return jsonify({
            'success': True,
            'city': city,
            'predictions': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/performance/<city>', methods=['GET'])
def get_performance(city):
    """Get model performance history"""
    try:
        performances = ModelPerformance.query.filter_by(
            city=city
        ).order_by(ModelPerformance.timestamp.desc()).limit(10).all()

        if not performances:
            return jsonify({
                'success': False,
                'error': 'No performance data found'
            })

        # Format the response
        result = []
        for perf in performances:
            result.append({
                'timestamp': perf.timestamp.isoformat(),
                'model_name': perf.model_name,
                'r2_score': perf.r2_score,
                'mae': perf.mae,
                'rmse': perf.rmse,
                'detailed_metrics': perf.detailed_metrics
            })

        return jsonify({
            'success': True,
            'city': city,
            'performances': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/debug-db')
def debug_db():
    """Debug database state"""
    try:
        # Check all predictions
        all_predictions = Prediction.query.all()
        predictions_data = []
        for p in all_predictions:
            predictions_data.append({
                'id': p.id,
                'city': p.city,
                'date': p.prediction_date.isoformat(),
                'generated': p.generation_timestamp.isoformat(),
                'is_current': p.is_current,
                'version': p.version,
                'model': p.model_version
            })

        # Check performance data
        all_performance = ModelPerformance.query.all()
        performance_data = []
        for p in all_performance:
            performance_data.append({
                'id': p.id,
                'city': p.city,
                'timestamp': p.timestamp.isoformat(),
                'model': p.model_name,
                'r2': p.r2_score,
                'has_detailed_metrics': bool(p.detailed_metrics)
            })

        return jsonify({
            'predictions': predictions_data,
            'performance': performance_data,
            'prediction_count': len(predictions_data),
            'performance_count': len(performance_data),
            'database_type': 'PostgreSQL' if 'postgresql' in app.config["SQLALCHEMY_DATABASE_URI"] else 'SQLite'
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/fix-metrics')
def fix_metrics():
    """Fix existing performance records with 0 R² scores"""
    try:
        performances = ModelPerformance.query.filter(
            ModelPerformance.r2_score == 0
        ).all()

        fixed_count = 0
        for perf in performances:
            if perf.detailed_metrics:
                r2_scores = []
                for param_metrics in perf.detailed_metrics.values():
                    if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                        r2_scores.append(param_metrics['r2'])
                if r2_scores:
                    perf.r2_score = sum(r2_scores) / len(r2_scores)
                    fixed_count += 1

        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Fixed {fixed_count} performance records'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/debug-metrics/<city>')
def debug_metrics(city):
    """Debug metrics for a city"""
    try:
        performances = ModelPerformance.query.filter_by(
            city=city
        ).order_by(ModelPerformance.timestamp.desc()).all()

        result = []
        for perf in performances:
            result.append({
                'timestamp': perf.timestamp.isoformat(),
                'model': perf.model_name,
                'r2_score': perf.r2_score,
                'has_detailed_metrics': bool(perf.detailed_metrics),
                'detailed_metrics_keys': list(perf.detailed_metrics.keys()) if perf.detailed_metrics else []
            })

        return jsonify({
            'success': True,
            'performances': result
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/historical-data/<city>', methods=['GET'])
def get_historical_data(city):
    """Get historical weather data for analysis"""
    try:
        # Get date range from query parameters or use default
        days = request.args.get('days', 30, type=int)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        # Get historical predictions
        predictions = Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= start_date,
            Prediction.prediction_date <= end_date,
            Prediction.is_current == True  # Only get the most recent predictions for each date
        ).order_by(Prediction.prediction_date.asc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found for this city'
            })

        # Format the response
        result = []
        for pred in predictions:
            result.append({
                'date': pred.prediction_date.isoformat(),
                'min_temp': pred.min_temp,
                'max_temp': pred.max_temp,
                'avg_temp': pred.avg_temp,
                'humidity': pred.humidity,
                'wind_speed': pred.wind_speed,
                'condition': pred.condition,
                'model_version': pred.model_version,
                'generated_at': pred.generation_timestamp.isoformat()
            })

        return jsonify({
            'success': True,
            'city': city,
            'data': result,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/historical-performance/<city>', methods=['GET'])
def get_historical_performance(city):
    """Get historical model performance data"""
    try:
        # Get limit from query parameters or use default
        limit = request.args.get('limit', 10, type=int)

        performances = ModelPerformance.query.filter_by(
            city=city
        ).order_by(ModelPerformance.timestamp.desc()).limit(limit).all()

        if not performances:
            return jsonify({
                'success': False,
                'error': 'No performance data found for this city'
            })

        # Format the response
        result = []
        for perf in performances:
            result.append({
                'timestamp': perf.timestamp.isoformat(),
                'model_name': perf.model_name,
                'r2_score': perf.r2_score,
                'mae': perf.mae,
                'rmse': perf.rmse,
                'detailed_metrics': perf.detailed_metrics
            })

        return jsonify({
            'success': True,
            'city': city,
            'performances': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


import csv
from io import StringIO
from flask import make_response


@app.route('/download-historical-data/<city>', methods=['GET'])
def download_historical_data(city):
    """Download historical weather data as CSV"""
    try:
        # Get days parameter or default to 30
        days = request.args.get('days', 30, type=int)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        # Get historical predictions
        predictions = Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= start_date,
            Prediction.prediction_date <= end_date,
            Prediction.is_current == True
        ).order_by(Prediction.prediction_date.asc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found for this city'
            })

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Date', 'Min Temperature (°C)', 'Max Temperature (°C)',
            'Average Temperature (°C)', 'Humidity (%)',
            'Wind Speed (km/h)', 'Weather Condition', 'Model Version',
            'Prediction Generated At'
        ])

        # Write data rows
        for pred in predictions:
            writer.writerow([
                pred.prediction_date.strftime('%Y-%m-%d'),
                pred.min_temp,
                pred.max_temp,
                pred.avg_temp,
                pred.humidity,
                pred.wind_speed,
                pred.condition,
                pred.model_version,
                pred.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename={city}_weather_data.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/download-performance-data/<city>', methods=['GET'])
def download_performance_data(city):
    """Download model performance data as CSV"""
    try:
        # Get limit parameter
        limit = request.args.get('limit', 50, type=int)

        # Get performance data
        if limit == 'all':
            performances = ModelPerformance.query.filter_by(city=city).all()
        else:
            performances = ModelPerformance.query.filter_by(
                city=city
            ).order_by(ModelPerformance.timestamp.desc()).limit(limit).all()

        if not performances:
            return jsonify({
                'success': False,
                'error': 'No performance data found for this city'
            })

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Timestamp', 'Model Name', 'R² Score', 'MAE', 'RMSE',
            'Avg Temp R²', 'Min Temp R²', 'Max Temp R²',
            'Humidity R²', 'Wind Speed R²'
        ])

        # Write data rows
        for perf in performances:
            metrics = perf.detailed_metrics or {}

            writer.writerow([
                perf.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                perf.model_name,
                perf.r2_score,
                perf.mae,
                perf.rmse,
                metrics.get('avg_c', {}).get('r2', 'N/A'),
                metrics.get('min_c', {}).get('r2', 'N/A'),
                metrics.get('max_c', {}).get('r2', 'N/A'),
                metrics.get('humidity', {}).get('r2', 'N/A'),
                metrics.get('wind_kph', {}).get('r2', 'N/A')
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename={city}_model_performance.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/download-all-weather-data/<city>', methods=['GET'])
def download_all_weather_data(city):
    """Download ALL historical weather data as CSV from database"""
    try:
        # Get ALL historical predictions for this city
        predictions = Prediction.query.filter(
            Prediction.city == city,
            Prediction.is_current == True
        ).order_by(Prediction.prediction_date.asc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found for this city'
            })

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Date', 'Min Temperature (°C)', 'Max Temperature (°C)',
            'Average Temperature (°C)', 'Humidity (%)',
            'Wind Speed (km/h)', 'Weather Condition', 'Model Version',
            'Prediction Generated At', 'Version'
        ])

        # Write data rows
        for pred in predictions:
            writer.writerow([
                pred.prediction_date.strftime('%Y-%m-%d'),
                pred.min_temp,
                pred.max_temp,
                pred.avg_temp,
                pred.humidity,
                pred.wind_speed,
                pred.condition,
                pred.model_version,
                pred.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                pred.version
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename={city}_all_weather_data.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download-all-performance-data/<city>', methods=['GET'])
def download_all_performance_data(city):
    """Download ALL model performance data as CSV from database"""
    try:
        # Get ALL performance data for this city
        performances = ModelPerformance.query.filter_by(city=city).order_by(ModelPerformance.timestamp.desc()).all()

        if not performances:
            return jsonify({
                'success': False,
                'error': 'No performance data found for this city'
            })

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Timestamp', 'Model Name', 'R² Score', 'MAE', 'RMSE',
            'Avg Temp R²', 'Min Temp R²', 'Max Temp R²',
            'Humidity R²', 'Wind Speed R²'
        ])

        # Write data rows
        for perf in performances:
            metrics = perf.detailed_metrics or {}

            writer.writerow([
                perf.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                perf.model_name,
                perf.r2_score,
                perf.mae,
                perf.rmse,
                metrics.get('avg_c', {}).get('r2', 'N/A'),
                metrics.get('min_c', {}).get('r2', 'N/A'),
                metrics.get('max_c', {}).get('r2', 'N/A'),
                metrics.get('humidity', {}).get('r2', 'N/A'),
                metrics.get('wind_kph', {}).get('r2', 'N/A')
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename={city}_all_model_performance.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/deploy-check')
def deploy_check():
    """Check if it's safe to deploy based on Jordan time - ONLY 3:00 AM and later"""
    try:
        from datetime import datetime
        import pytz
        
        # Get current time in Jordan
        jordan_tz = pytz.timezone('Asia/Amman')
        jordan_time = datetime.now(jordan_tz)
        current_hour = jordan_time.hour
        current_minute = jordan_time.minute
        
        # Safe to deploy ONLY from 3:00 AM Jordan time onwards
        is_safe_time = current_hour >= 3
        
        # More specific messages
        if current_hour == 3 and current_minute == 0:
            message = 'PERFECT! It\'s exactly 3:00 AM Jordan time - SAFE TO DEPLOY!'
        elif current_hour >= 3:
            message = f'SAFE TO DEPLOY - It\'s {current_hour:02d}:{current_minute:02d} in Jordan (3:00 AM or later)'
        else:
            hours_left = 2 - current_hour
            minutes_left = 60 - current_minute
            if current_hour == 2:
                message = f'Wait {minutes_left} minutes until 3:00 AM Jordan time'
            else:
                message = f'Wait {hours_left} hours and {minutes_left} minutes until 3:00 AM Jordan time'
        
        return jsonify({
            'safe_to_deploy': is_safe_time,
            'jordan_time': jordan_time.strftime('%Y-%m-%d %H:%M:%S'),
            'utc_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'current_hour': current_hour,
            'current_minute': current_minute,
            'message': message,
            'database_type': 'PostgreSQL' if 'postgresql' in app.config["SQLALCHEMY_DATABASE_URI"] else 'SQLite'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/init-db')
def init_database():
    """Initialize database tables - RUN THIS FIRST"""
    try:
        with app.app_context():
            # Drop all tables if they exist
            db.drop_all()
            # Create all tables
            db.create_all()
            
            print("Database tables created successfully!")
            print("Tables: predictions, model_performance")
            
            return jsonify({
                'success': True,
                'message': 'Database initialized successfully!',
                'tables_created': ['predictions', 'model_performance'],
                'database_type': 'PostgreSQL'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})










@app.route('/fix-today-predictions')
def fix_today_predictions():
    """Force generate predictions for today and fix any date issues"""
    try:
        today = datetime.now().date()
        city = "Amman"
        
        print(f"Force generating predictions for {today}")
        
        # Mark all current predictions as not current
        reset_count = Prediction.query.filter(
            Prediction.is_current == True
        ).update({'is_current': False}, synchronize_session=False)
        
        db.session.commit()
        print(f"Reset {reset_count} predictions")
        
        # Force new training and prediction
        training_result = predictor.train_model(city)
        if 'error' in training_result:
            return jsonify({'error': training_result['error']})
            
        prediction_result = predictor.predict_weather(days=7)
        if 'error' in prediction_result:
            return jsonify({'error': prediction_result['error']})
        
        # Save to database
        model_metrics = {}
        if hasattr(predictor, 'results') and predictor.best_model_name:
            best_model_result = predictor.results[predictor.best_model_name]
            model_metrics = best_model_result.get('detailed_metrics', {})
        
        save_success = save_predictions_to_db(
            city,
            prediction_result['predictions'],
            predictor.best_model_name,
            model_metrics
        )
        
        if save_success:
            # Verify the new predictions
            new_predictions = get_latest_predictions_from_db(city, 7)
            if new_predictions:
                dates = [p['date'].strftime('%Y-%m-%d') for p in new_predictions]
                return jsonify({
                    'success': True,
                    'message': f'Generated new predictions starting from {today}',
                    'prediction_dates': dates,
                    'today': today.strftime('%Y-%m-%d')
                })
        
        return jsonify({'error': 'Failed to generate new predictions'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Start keep-alive system when app loads
keep_alive_manager.start_keep_alive()

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created/verified")
            print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        except Exception as e:
            print(f"Error creating tables: {e}")
            print(" Tables might already exist, continuing...")
    app.run(debug=True)
