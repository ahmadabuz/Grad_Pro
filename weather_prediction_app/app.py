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
import csv
from io import StringIO
from flask import make_response

app = Flask(__name__)

# POSTGRESQL DATABASE CONFIGURATION - THIS WILL PERSIST!
def get_database_uri():
    if 'RENDER' in os.environ:
        # use Render's PostgreSQL - THIS PERSISTS BETWEEN DEPLOYS
        database_url = os.environ.get('DATABASE_URL')
        if database_url and database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        print(f"Using PostgreSQL database: {database_url}")
        return database_url
    else:
        # fallback to sqlit (not efficient , as it will lose all new data if you deployed after making new commmit )
        return "sqlite:///weather_predictions.db"

app.config["SQLALCHEMY_DATABASE_URI"] = get_database_uri()
app.config["SQLALCHEMY_ECHO"] = False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
    model_comparison = db.Column(db.JSON, nullable=True)  # Store all model results
    

API_KEY = "923f98fb4a5d4421adf171802252711"
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
        self.city_climate_zones = {
            # Desert/Arid Climate
            'amman': 'desert', 'damascus': 'desert', 'cairo': 'desert',
            'dubai': 'desert', 'riyadh': 'desert', 'kuwait': 'desert',
            # Mediterranean Climate
            'beirut': 'mediterranean', 'jerusalem': 'mediterranean',
            'istanbul': 'mediterranean', 'athens': 'mediterranean',
            # Continental Climate
            'london': 'temperate', 'paris': 'temperate', 'berlin': 'temperate',
            'moscow': 'continental', 'warsaw': 'continental',
            # Tropical Climate
            'mumbai': 'tropical', 'bangkok': 'tropical', 'singapore': 'tropical',
            'manila': 'tropical',
            # Default
            'default': 'temperate'
        }
        self.current_city = None

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
            # Limit cache size to prevent memory issues
            if len(self.cache) > 50:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = prediction_data

    def clear_old_cache(self):
        """Clear cache entries"""
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

                # Check if city is invalid
                if response.status_code != 200:
                    try:
                        error_message = response.json().get('error', {}).get('message', 'Unknown error')
                        raise ValueError(f"WeatherAPI error: {error_message}")
                    except Exception:
                        raise ValueError("WeatherAPI returned invalid response.")

                data = response.json()

                # Ensure forecast data exists
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

    def get_climate_zone(self, city):
        """Determine climate zone for a city"""
        city_lower = city.lower()
        for city_key, zone in self.city_climate_zones.items():
            if city_key in city_lower:
                return zone
        return self.city_climate_zones['default']

    def prepare_advanced_features(self, data):
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        # Basic temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['date'].dt.weekday >= 5
        # WIND-SPECIFIC FEATURE ENGINEERING
        # Wind has different patterns than temperature - more random, less seasonal
        df['wind_seasonal_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 182.625)  # 6-month cycle for wind
        df['wind_daily_cycle'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25 * 4)  # Quarterly variations
        # Wind persistence features
        df['wind_persistence'] = df['wind_kph'].shift(1) / (df['wind_kph'] + 0.1)
        df['wind_change'] = df['wind_kph'].diff()
        df['wind_change_abs'] = abs(df['wind_change'])
        # Wind volatility features 
        for window in [3, 7, 14]:
            df[f'wind_volatility_{window}'] = df['wind_kph'].rolling(window=window, min_periods=1).std()
            df[f'wind_change_std_{window}'] = df['wind_change'].rolling(window=window, min_periods=1).std()
            
        for lag in [1, 2, 3]:  
            df[f'wind_kph_lag_{lag}'] = df['wind_kph'].shift(lag)
            df[f'wind_change_lag_{lag}'] = df['wind_change'].shift(lag)
            
        for lag in [1, 2, 3, 5, 7, 10, 14]:
            for col in ['avg_c', 'humidity', 'precip_mm']:  
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        for window in [3, 5, 7]:
            df[f'wind_kph_rolling_mean_{window}'] = df['wind_kph'].rolling(window=window, min_periods=1).mean()
            df[f'wind_kph_rolling_std_{window}'] = df['wind_kph'].rolling(window=window, min_periods=1).std()
            df[f'wind_kph_rolling_min_{window}'] = df['wind_kph'].rolling(window=window, min_periods=1).min()
            df[f'wind_kph_rolling_max_{window}'] = df['wind_kph'].rolling(window=window, min_periods=1).max()

        for window in [7, 10, 14]:
            for col in ['avg_c', 'humidity', 'precip_mm']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        # Pressure gradient simulation (wind driver)
        df['temp_gradient'] = df['max_c'] - df['min_c']  # Larger gradient = more wind
        df['humidity_pressure_effect'] = (100 - df['humidity']) * 0.1  # Lower humidity = higher pressure potential
        # Wind-specific interaction features
        df['temp_wind_interaction'] = df['avg_c'] * df['wind_kph']
        df['seasonal_wind_boost'] = df['wind_seasonal_factor'] * df['wind_kph']
        # Weather system features
        df['weather_system_change'] = (df['avg_c'].diff(1) + df['humidity'].diff(1) + df['wind_kph'].diff(1)).abs()
         # Fill missing values with wind-appropriate methods
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'wind' in col:
                # For wind features, use forward fill then random noise
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                if df[col].isna().any():
                    df[col] = df[col].fillna(np.random.uniform(5, 15))  # Reasonable wind range
            else:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
        return df    



            
    
    

    def calculate_heat_index(self, temp, humidity):
        """Calculate heat index for realistic temperature perception"""
        # Simplified heat index calculation
        return temp + 0.1 * (humidity - 50)

    def calculate_dew_point(self, temp, humidity):
        """Calculate dew point for humidity analysis"""
        # Magnus formula approximation
        alpha = 17.27
        beta = 237.7
        gamma = (alpha * temp) / (beta + temp) + np.log(humidity / 100.0)
        return (beta * gamma) / (alpha - gamma)

    def train_advanced_models(self, X_train, X_test, y_train, y_test):
    """Enhanced model training with wind-focused evaluation"""
    models = {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
        'Ridge': Ridge(alpha=1.0, random_state=42, max_iter=5000),
        'Lasso': Lasso(alpha=0.05, random_state=42, max_iter=5000),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        try:
            if name in ['ElasticNet', 'Ridge', 'Lasso', 'GradientBoosting', 'XGBoost', 'LightGBM']:
                multi_model = MultiOutputRegressor(model, n_jobs=1)
                multi_model.fit(X_train, y_train)
                y_pred_train = multi_model.predict(X_train)
                y_pred = multi_model.predict(X_test)
                trained_model = multi_model
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred = model.predict(X_test)
                trained_model = model

            # Calculate metrics with WIND-SPECIFIC evaluation
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred)

            print(f"{name}: R² Train = {r2_train:.4f}, R² Test = {r2_test:.4f}")

            # Calculate metrics for each target - SPECIAL ATTENTION TO WIND
            metrics = {}
            target_names = ['avg_c', 'min_c', 'max_c', 'humidity', 'wind_kph']
            
            for i, target_name in enumerate(target_names):
                y_true_target = y_test[:, i]
                y_pred_target = y_pred[:, i]
                
                metrics[target_name] = {
                    'mae': mean_absolute_error(y_true_target, y_pred_target),
                    'rmse': np.sqrt(mean_squared_error(y_true_target, y_pred_target)),
                    'r2': r2_score(y_true_target, y_pred_target)
                }
                
                # Special wind validation
                if target_name == 'wind_kph':
                    wind_metrics = self._validate_wind_predictions(y_true_target, y_pred_target)
                    metrics[target_name].update(wind_metrics)

            # Overall metrics with wind quality check
            overall_metrics = {
                'mae': np.mean([m['mae'] for m in metrics.values()]),
                'rmse': np.mean([m['rmse'] for m in metrics.values()]),
                'r2': np.mean([m['r2'] for m in metrics.values()]),
                'model': trained_model,
                'detailed_metrics': metrics,
                'wind_quality_score': metrics['wind_kph'].get('quality_score', 0)
            }

            results[name] = overall_metrics

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    return results

def _validate_wind_predictions(self, y_true, y_pred):
    """Validate that wind predictions are physically realistic"""
    # Check for zero winds
    zero_wind_count = np.sum(y_pred < 1.0)
    zero_wind_ratio = zero_wind_count / len(y_pred)
    
    # Check wind range
    realistic_range_ratio = np.sum((y_pred >= 2.0) & (y_pred <= 50.0)) / len(y_pred)
    
    # Check variance (wind should have some variation)
    pred_variance = np.var(y_pred)
    true_variance = np.var(y_true)
    variance_ratio = pred_variance / (true_variance + 1e-8)
    
    quality_score = (
        (1 - zero_wind_ratio) * 0.4 +
        realistic_range_ratio * 0.3 +
        min(1.0, variance_ratio) * 0.3
    )
    
    return {
        'zero_wind_ratio': zero_wind_ratio,
        'realistic_range_ratio': realistic_range_ratio,
        'variance_ratio': variance_ratio,
        'quality_score': quality_score
    }

    def add_realistic_variation(self, predictions, city):
        """Add realistic weather variations based on climate zone"""
        climate_zone = self.get_climate_zone(city)

        for i, pred in enumerate(predictions):
            # Add small random variations based on climate zone
            variation_factors = {
                'desert': {'temp_var': 1.5, 'humidity_var': 3, 'wind_var': 2},
                'mediterranean': {'temp_var': 2.0, 'humidity_var': 5, 'wind_var': 3},
                'temperate': {'temp_var': 2.5, 'humidity_var': 8, 'wind_var': 4},
                'continental': {'temp_var': 3.0, 'humidity_var': 10, 'wind_var': 5},
                'tropical': {'temp_var': 1.0, 'humidity_var': 5, 'wind_var': 3}
            }

            factors = variation_factors.get(climate_zone, variation_factors['temperate'])

            # Add progressive variation (more variation in later days)
            progress_factor = (i + 1) / len(predictions)

            # Temperature variation
            temp_variation = np.random.normal(0, factors['temp_var'] * progress_factor)
            pred['min_temp'] = round(pred['min_temp'] + temp_variation * 0.7, 1)
            pred['max_temp'] = round(pred['max_temp'] + temp_variation * 1.3, 1)
            pred['avg_temp'] = round((pred['min_temp'] + pred['max_temp']) / 2, 1)

            # Humidity variation
            humidity_variation = np.random.normal(0, factors['humidity_var'] * progress_factor)
            pred['humidity'] = round(max(10, min(95, pred['humidity'] + humidity_variation)), 1)

            # Wind variation
            wind_variation = np.random.normal(0, factors['wind_var'] * progress_factor)
            pred['wind'] = round(max(0, pred['wind'] + wind_variation), 1)

            # Ensure logical relationships
            if pred['min_temp'] > pred['avg_temp']:
                pred['avg_temp'] = (pred['min_temp'] + pred['max_temp']) / 2
            if pred['avg_temp'] > pred['max_temp']:
                pred['avg_temp'] = (pred['min_temp'] + pred['max_temp']) / 2

        return predictions

    def create_future_features(self, last_date, future_days, df, feature_cols, best_model, scaler_X, scaler_y):
    """Enhanced future feature creation with wind physics"""
    future_features_list = []
    last_features = df[feature_cols].iloc[-1:].to_dict('records')[0]
    recent_data = df[['avg_c', 'humidity', 'wind_kph', 'precip_mm']].tail(30).to_dict('records')
    
    climate_zone = self.get_climate_zone(getattr(self, 'current_city', 'amman'))
    
    # Climate-specific wind characteristics
    wind_profiles = {
        'desert': {'base': 8.0, 'variation': 4.0, 'seasonal_amp': 2.0},
        'mediterranean': {'base': 12.0, 'variation': 6.0, 'seasonal_amp': 3.0},
        'temperate': {'base': 15.0, 'variation': 8.0, 'seasonal_amp': 4.0},
        'continental': {'base': 10.0, 'variation': 5.0, 'seasonal_amp': 3.0},
        'tropical': {'base': 8.0, 'variation': 4.0, 'seasonal_amp': 2.0}
    }
    wind_profile = wind_profiles.get(climate_zone, wind_profiles['temperate'])

    for i in range(future_days):
        future_date = last_date + timedelta(days=i + 1)
        day_of_year = future_date.timetuple().tm_yday

        current_features = last_features.copy()
        
        # Time features
        current_features['day_of_year'] = day_of_year
        current_features['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        current_features['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        current_features['month'] = future_date.month
        current_features['week_of_year'] = future_date.isocalendar().week
        current_features['is_weekend'] = future_date.weekday() >= 5
        
        # WIND-SPECIFIC TIME FEATURES
        current_features['wind_seasonal_factor'] = np.sin(2 * np.pi * day_of_year / 182.625)
        current_features['wind_daily_cycle'] = np.sin(2 * np.pi * day_of_year / 365.25 * 4)
        
        # Seasonal wind patterns
        seasonal_multiplier = 1.0
        if future_date.month in [11, 12, 1, 2]:  # Winter - typically windier
            seasonal_multiplier = 1.3
        elif future_date.month in [5, 6, 7, 8]:  # Summer - variable
            seasonal_multiplier = 0.9 if climate_zone == 'desert' else 1.1
            
        # Generate realistic wind base with physics
        base_wind = wind_profile['base'] * seasonal_multiplier
        seasonal_variation = wind_profile['seasonal_amp'] * current_features['wind_seasonal_factor']
        random_variation = np.random.normal(0, wind_profile['variation'])
        
        # Combine components for realistic wind
        realistic_wind = max(3.0, base_wind + seasonal_variation + random_variation)
        
        # Update wind features with realistic patterns
        self._update_wind_features(current_features, realistic_wind, recent_data, feature_cols)
        
        # Convert to DataFrame for prediction
        current_features_df = pd.DataFrame([current_features])[feature_cols]
        current_features_scaled = scaler_X.transform(current_features_df.values)
        predicted_scaled = best_model.predict(current_features_scaled)
        predicted_original = scaler_y.inverse_transform(predicted_scaled)

        # Store prediction and update recent data
        predicted_data = {
            'avg_c': predicted_original[0][0],
            'min_c': predicted_original[0][1],
            'max_c': predicted_original[0][2],
            'humidity': predicted_original[0][3],
            'wind_kph': max(3.0, predicted_original[0][4])  # Logical minimum
        }
        recent_data.append(predicted_data)
        recent_data = recent_data[-30:]
        
        # Update features for next iteration with wind-aware propagation
        self._propagate_features(current_features, predicted_data, recent_data, feature_cols, climate_zone)
        
        future_features_list.append(pd.DataFrame([current_features])[feature_cols].values.flatten())
        last_features = current_features

    return np.array(future_features_list)

def _update_wind_features(self, features, realistic_wind, recent_data, feature_cols):
    """Update wind-specific features with realistic patterns"""
    # Update wind lag features
    for lag in [1, 2, 3]:
        wind_lag_col = f'wind_kph_lag_{lag}'
        if wind_lag_col in features:
            if len(recent_data) > lag:
                # Use actual recent data when available
                features[wind_lag_col] = recent_data[-lag]['wind_kph']
            else:
                # Initialize with realistic wind
                features[wind_lag_col] = realistic_wind * (0.8 + 0.4 * np.random.random())
    
    # Update wind rolling features
    temp_df = pd.DataFrame(recent_data)
    for window in [3, 5, 7]:
        for stat in ['mean', 'std', 'min', 'max']:
            wind_roll_col = f'wind_kph_rolling_{stat}_{window}'
            if wind_roll_col in features and len(temp_df) >= window:
                if stat == 'mean':
                    value = temp_df['wind_kph'].rolling(window=window).mean().iloc[-1]
                elif stat == 'std':
                    value = temp_df['wind_kph'].rolling(window=window).std().iloc[-1]
                elif stat == 'min':
                    value = temp_df['wind_kph'].rolling(window=window).min().iloc[-1]
                else:  # max
                    value = temp_df['wind_kph'].rolling(window=window).max().iloc[-1]
                
                if not pd.isna(value):
                    features[wind_roll_col] = value
    
    # Update wind volatility features
    if len(recent_data) > 1:
        wind_changes = [recent_data[i]['wind_kph'] - recent_data[i-1]['wind_kph'] 
                       for i in range(1, len(recent_data))]
        if wind_changes:
            features['wind_change'] = wind_changes[-1]
            features['wind_change_abs'] = abs(wind_changes[-1])

def _propagate_features(self, features, predicted_data, recent_data, feature_cols, climate_zone):
    """Propagate features for next time step with wind awareness"""
    temp_df = pd.DataFrame(recent_data)
    
    # Update temperature and humidity features (your existing logic)
    for col in ['avg_c', 'humidity']:
        for lag in [1, 2, 3, 7]:
            lag_col = f'{col}_lag_{lag}'
            if lag_col in features and len(temp_df) > lag:
                lag_value = temp_df[col].shift(lag).iloc[-1]
                if not pd.isna(lag_value):
                    features[lag_col] = lag_value
    
    # Update interaction features
    features['temp_wind_interaction'] = predicted_data['avg_c'] * predicted_data['wind_kph']
    features['temp_gradient'] = predicted_data['max_c'] - predicted_data['min_c']
    def predict_weather(self, days=7):
        """Generate weather predictions with enhanced realism"""
        try:
            if self.model is None:
                return {"error": "Model not trained"}

            today = datetime.now().date()
            print(f"Generating predictions starting from: {today}")

            # Get base predictions
            future_X = self.create_future_features(
                pd.to_datetime(today),
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

            # Apply realistic constraints
            predictions_df['humidity'] = predictions_df['humidity'].clip(10, 95)
            predictions_df['min_temp'] = predictions_df['min_temp'].clip(-50, 50)
            predictions_df['max_temp'] = predictions_df['max_temp'].clip(-50, 50)
            predictions_df['avg_temp'] = predictions_df['avg_temp'].clip(-50, 50)
            predictions_df['wind'] = predictions_df['wind'].clip(0, 150)

            # Ensure logical temperature relationships
            for i, row in predictions_df.iterrows():
                if row['min_temp'] > row['avg_temp']:
                    predictions_df.at[i, 'avg_temp'] = (row['min_temp'] + row['max_temp']) / 2
                if row['avg_temp'] > row['max_temp']:
                    predictions_df.at[i, 'avg_temp'] = (row['min_temp'] + row['max_temp']) / 2

            # Generate dates
            future_dates = [today + timedelta(days=i) for i in range(days)]
            predictions_df['date'] = future_dates
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])

            if 'precip_mm' in self.df.columns:
                recent_precip = self.df['precip_mm'].tail(30).mean()
            else:
                recent_precip = 0

            # Convert to list format
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

            # Apply climate-specific variations
            if hasattr(self, 'current_city') and self.current_city:
                predictions_list = self.add_realistic_variation(predictions_list, self.current_city)

            print(f"Generated predictions for: {[d.strftime('%Y-%m-%d') for d in future_dates]}")
            return {"success": True, "predictions": predictions_list}

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}

    def train_model(self, city):
        """Complete model training pipeline with city context"""
        try:
            self.current_city = city  # Store city for variation adjustments

            # Fetch data
            data = self.fetch_weather_data(city, days=365)
            if not data or len(data) < 50:
                return {"error": f"No city found with name '{city}' or insufficient data."}

            # Prepare features
            self.df = self.prepare_advanced_features(data)

            # Target columns
            target_cols = ['avg_c', 'min_c', 'max_c', 'humidity', 'wind_kph']
            self.feature_cols = [col for col in self.df.columns if col not in target_cols + ['date']]

            X = self.df[self.feature_cols].values
            y = self.df[target_cols].values

            # Enhanced train-test split with time series awareness
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features
            self.scaler_X = StandardScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)

            self.scaler_y = StandardScaler()
            y_train_scaled = self.scaler_y.fit_transform(y_train)
            y_test_scaled = self.scaler_y.transform(y_test)

            # Train models
            self.results = self.train_advanced_models(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)

            if not self.results:
                return {"error": "No models trained successfully"}

            # Find best model
            best_r2 = -np.inf
            for name, result in self.results.items():
                if result['r2'] > best_r2:
                    best_r2 = result['r2']
                    self.best_model_name = name

            # Use the best model
            self.model = self.results[self.best_model_name]['model']

            print(f"Best model: {self.best_model_name} with R²: {best_r2:.4f}")
            return {"success": True, "best_model": self.best_model_name, "r2_score": best_r2}

        except Exception as e:
            return {"error": str(e)}

    def determine_weather_condition(self, avg_temp, min_temp, max_temp, humidity, wind, month, recent_precip,
                                    day_index):
        """Enhanced weather condition determination with more nuanced descriptions"""

        # Get climate zone for more accurate conditions
        climate_zone = self.get_climate_zone(getattr(self, 'current_city', 'amman'))

        # Temperature classification with climate context
        temp_range = max_temp - min_temp

        if climate_zone == 'desert':
            # Desert climate adjustments
            if avg_temp > 28:
                temp_intensity = "Hot"
                temp_feel = "warm" if avg_temp < 32 else "hot"
            elif avg_temp > 22:
                temp_intensity = "Pleasantly Warm"
                temp_feel = "mild"
            elif avg_temp > 16:
                temp_intensity = "Mild"
                temp_feel = "cool"
            else:
                temp_intensity = "Cool"
                temp_feel = "chilly"
        else:
            # Standard classification for other climates
            if avg_temp > 25:
                temp_intensity = "Warm"
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
            else:
                temp_intensity = "Cold"
                temp_feel = "cold"

        # Enhanced cloud cover based on humidity and climate
        cloud_cover = ""
        precipitation = ""
        special_conditions = []

        # Desert-specific cloud logic
        if climate_zone == 'desert':
            if humidity > 40:
                cloud_cover = "Partly Cloudy"
                if humidity > 60 and avg_temp > 20:
                    precipitation = "Isolated Showers"
            elif humidity > 25:
                cloud_cover = "Mostly Clear"
            else:
                cloud_cover = "Clear"
        else:
            # Standard cloud logic
            if humidity > 75:
                cloud_cover = "Mostly Cloudy"
            elif humidity > 60:
                cloud_cover = "Partly Cloudy"
            elif humidity > 45:
                cloud_cover = "Mostly Clear"
            else:
                cloud_cover = "Clear"

        # Wind description with climate context
        wind_description = ""
        if wind > 25:
            wind_description = "Windy"
            special_conditions.append("Breezy Conditions")
        elif wind > 15:
            wind_description = "Moderate Breeze"
        elif wind > 8:
            wind_description = "Light Breeze"
        else:
            wind_description = "Calm"

        # Temperature variation analysis
        if temp_range > 12:
            special_conditions.append("Large Daily Temperature Range")
        elif temp_range > 8:
            special_conditions.append("Noticeable Temperature Swing")

        # Seasonal and climate-specific conditions
        is_fall = month in [9, 10, 11]
        is_spring = month in [3, 4, 5]

        if climate_zone == 'desert':
            if humidity < 25:
                special_conditions.append("Dry Conditions")
            if temp_range > 10:
                special_conditions.append("Desert Climate Pattern")

        if is_fall and day_index > 2:
            special_conditions.append("Autumn Weather")
        elif is_spring and day_index < 4:
            special_conditions.append("Springlike Conditions")

        # Visibility and atmospheric conditions
        if humidity < 30:
            special_conditions.append("Excellent Visibility")
        elif humidity > 70:
            special_conditions.append("Reduced Visibility")

        if wind < 5 and cloud_cover == "Clear":
            special_conditions.append("Calm and Clear")

        # Build condition description
        condition_parts = [temp_intensity]

        # Add cloud cover
        condition_parts.append(cloud_cover)

        # Add precipitation if any
        if precipitation:
            condition_parts.append(f"with {precipitation}")

        # Add wind if significant
        if wind > 15:
            condition_parts.append(wind_description)

        # Add special conditions (limit to 2 most relevant)
        relevant_conditions = []
        for condition in special_conditions:
            if "Temperature" in condition or "Visibility" in condition or climate_zone in condition:
                relevant_conditions.append(condition)

        # Add up to 2 most relevant special conditions
        condition_parts.extend(relevant_conditions[:2])

        # Add temperature feel
        condition_parts.append(f"Feeling {temp_feel}")

        # Create final description
        detailed_condition = ", ".join(condition_parts)

        # Clean up and ensure proper formatting
        detailed_condition = detailed_condition.replace(" ,", ",")
        detailed_condition = " ".join(detailed_condition.split())

        # Ensure reasonable length
        if len(detailed_condition) > 100:
            parts = detailed_condition.split(", ")
            # Keep temperature, cloud cover, and 1-2 most important conditions
            essential_parts = parts[:3] + [parts[-1]]  # Keep first 3 and feeling
            detailed_condition = ", ".join(essential_parts)

        return detailed_condition.title()

    def generate_chart(self, predictions):
        """Generate weather prediction chart"""
        try:
            # Extract data for plotting
            dates = [pred['date'] for pred in predictions]
            avg_temps = [pred['avg_temp'] for pred in predictions]
            min_temps = [pred['min_temp'] for pred in predictions]
            max_temps = [pred['max_temp'] for pred in predictions]
            humidity = [pred['humidity'] for pred in predictions]
            wind = [pred['wind'] for pred in predictions]

            # Create the plot
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Temperature plot
            axes[0].plot(dates, avg_temps, 'r-o', label='Average', linewidth=2, markersize=6)
            axes[0].fill_between(dates, min_temps, max_temps, alpha=0.3, color='lightblue', label='Min-Max Range')
            axes[0].set_title('Temperature Forecast', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

            # Humidity plot
            axes[1].plot(dates, humidity, 'b-s', linewidth=2, markersize=6)
            axes[1].set_title('Humidity Forecast', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Humidity (%)')
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

            # Wind plot
            axes[2].plot(dates, wind, 'g-^', linewidth=2, markersize=6)
            axes[2].set_title('Wind Speed Forecast', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Wind Speed (kph)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
            plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Convert plot to base64 string
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

def save_predictions_to_db(city, predictions, model_name, model_metrics, model_comparison=None):
    """Save predictions to the database, properly managing is_current flags"""
    try:
        generation_time = datetime.now()
        today = datetime.now().date()

        # FIXED: Mark ALL previous predictions for this city as not current
        # (not just future dates)
        Prediction.query.filter(
            Prediction.city == city,
            Prediction.is_current == True
        ).update({'is_current': False}, synchronize_session=False)

        # Get the maximum version number for this city to increment it
        max_version = db.session.query(db.func.max(Prediction.version)).filter_by(
            city=city
        ).scalar() or 0

        new_version = max_version + 1

        for prediction in predictions:
            # Convert string date to datetime object if needed
            if isinstance(prediction['date'], str):
                prediction_date = datetime.strptime(prediction['date'], '%Y-%m-%d').date()
            else:
                prediction_date = prediction['date'].date()

            # Only save predictions for today and future dates as current
            # Past dates should remain as historical records
            is_current_prediction = prediction_date >= today

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
                is_current=is_current_prediction,  # Only current for today/future
                version=new_version
            )
            db.session.add(new_prediction)

        # Calculate overall R² score from detailed metrics
        overall_r2 = 0
        overall_mae = 0
        overall_rmse = 0

        if model_metrics and isinstance(model_metrics, dict):
            # Extract R² scores from each parameter and average them
            r2_scores = []
            mae_scores = []
            rmse_scores = []
            for param_metrics in model_metrics.values():
                if isinstance(param_metrics, dict):
                    if 'r2' in param_metrics:
                        r2_scores.append(param_metrics['r2'])
                    if 'mae' in param_metrics:
                        mae_scores.append(param_metrics['mae'])
                    if 'rmse' in param_metrics:
                        rmse_scores.append(param_metrics['rmse'])

            if r2_scores:
                overall_r2 = sum(r2_scores) / len(r2_scores)
            if mae_scores:
                overall_mae = sum(mae_scores) / len(mae_scores)
            if rmse_scores:
                overall_rmse = sum(rmse_scores) / len(rmse_scores)

        # Always save performance metrics with model comparison data
        performance = ModelPerformance(
            city=city,
            timestamp=generation_time,
            model_name=model_name or "Unknown",
            r2_score=round(overall_r2, 4),
            mae=round(overall_mae, 4),
            rmse=round(overall_rmse, 4),
            detailed_metrics=model_metrics,
            model_comparison=model_comparison  # Store complete model comparison
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
            model_results = {}

            # Try to get performance metrics with model comparison
            latest_performance = ModelPerformance.query.filter_by(
                city=city
            ).order_by(ModelPerformance.timestamp.desc()).first()

            if latest_performance:
                # Handle both old and new metric formats
                metrics = latest_performance.detailed_metrics or {}
                model_name = latest_performance.model_name or "Unknown"
                r2_score = latest_performance.r2_score or 0
                
                # Get model comparison data if available
                if latest_performance.model_comparison:
                    model_results = latest_performance.model_comparison
                else:
                    # Fallback: create basic model comparison from available data
                    if metrics:
                        # Calculate average metrics for the best model
                        r2_scores = []
                        mae_scores = []
                        rmse_scores = []
                        
                        for param_metrics in metrics.values():
                            if isinstance(param_metrics, dict):
                                if 'r2' in param_metrics:
                                    r2_scores.append(param_metrics['r2'])
                                if 'mae' in param_metrics:
                                    mae_scores.append(param_metrics['mae'])
                                if 'rmse' in param_metrics:
                                    rmse_scores.append(param_metrics['rmse'])
                        
                        if r2_scores:
                            model_results[model_name] = {
                                'r2': sum(r2_scores) / len(r2_scores),
                                'mae': sum(mae_scores) / len(mae_scores) if mae_scores else 0,
                                'rmse': sum(rmse_scores) / len(rmse_scores) if rmse_scores else 0
                            }

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
                'model_results': model_results,  # Include model comparison data
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

        # Safely get model metrics and results
        model_metrics = {}
        overall_r2 = 0
        model_results = {}

        if (hasattr(predictor, 'results') and predictor.results):
            # Extract all model results for comparison
            for model_name, result in predictor.results.items():
                model_results[model_name] = {
                    'r2': round(result.get('r2', 0), 4),
                    'mae': round(result.get('mae', 0), 4),
                    'rmse': round(result.get('rmse', 0), 4)
                }

            # Get best model metrics
            if predictor.best_model_name and predictor.best_model_name in predictor.results:
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

        # Save to database with model comparison data
        save_success = save_predictions_to_db(
            city,
            prediction_result['predictions'],
            predictor.best_model_name,
            metrics_dict,
            model_results  # Pass model comparison data
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
            'model_results': model_results,  # Include model comparison data
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
    """Get historical weather data - PAST DATES INCLUDING TODAY"""
    try:
        today = datetime.now().date()
        print(f"Historical data request for {city} - returning data up to {today}")

        # Get predictions for PAST dates INCLUDING TODAY, latest version only
        from sqlalchemy import func

        # Subquery to get the latest generation_timestamp for each date
        subquery = db.session.query(
            Prediction.prediction_date,
            func.max(Prediction.generation_timestamp).label('max_timestamp')
        ).filter(
            Prediction.city == city,
            Prediction.prediction_date <= today  # DATES UP TO AND INCLUDING TODAY
        ).group_by(Prediction.prediction_date).subquery()

        # Get predictions with the latest timestamp for each date
        predictions = Prediction.query.join(
            subquery,
            (Prediction.prediction_date == subquery.c.prediction_date) &
            (Prediction.generation_timestamp == subquery.c.max_timestamp)
        ).order_by(Prediction.prediction_date.desc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': f'No historical data found for city "{city}"'
            })

        print(f"Returning {len(predictions)} historical date predictions for {city} (up to {today})")

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
                'generated_at': pred.generation_timestamp.isoformat(),
                'version': pred.version,
                'is_current': pred.is_current
            })

        return jsonify({
            'success': True,
            'city': city,
            'data': result,
            'total_days': len(result),
            'cutoff_date': today.isoformat(),
            'note': f'Returning {len(result)} historical days (up to {today})'
        })
    except Exception as e:
        print(f"Error in historical data: {e}")
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



@app.route('/download-historical-data/<city>', methods=['GET'])
def download_historical_data(city):
    """Download historical weather data - PAST DATES INCLUDING TODAY"""
    try:
        days = request.args.get('days', 30, type=int)
        today = datetime.now().date()
        
        print(f"Download request for {city}: {days} days up to {today}")

        # Get the latest version for each PAST date (including today)
        from sqlalchemy import func
        
        subquery = db.session.query(
            Prediction.prediction_date,
            func.max(Prediction.generation_timestamp).label('max_timestamp')
        ).filter(
            Prediction.city == city,
            Prediction.prediction_date <= today
        ).group_by(Prediction.prediction_date).subquery()

        all_predictions = Prediction.query.join(
            subquery,
            (Prediction.prediction_date == subquery.c.prediction_date) &
            (Prediction.generation_timestamp == subquery.c.max_timestamp)
        ).order_by(Prediction.prediction_date.desc()).limit(days * 2).all()

        if not all_predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found for this city'
            })

        # Take the most recent 'days' predictions
        if len(all_predictions) < days:
            selected_predictions = all_predictions
        else:
            selected_predictions = all_predictions[:days]

        # Sort chronologically for CSV
        selected_predictions.sort(key=lambda x: x.prediction_date)

        print(f"Download prepared for {city}: {len(selected_predictions)} historical days")

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Date', 'Min Temperature (°C)', 'Max Temperature (°C)',
            'Average Temperature (°C)', 'Humidity (%)',
            'Wind Speed (km/h)', 'Weather Condition', 'Model Version',
            'Prediction Generated At', 'Version', 'Is Current'
        ])

        # Write data rows
        for pred in selected_predictions:
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
                pred.version,
                'Yes' if pred.is_current else 'No'
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        actual_days = len(selected_predictions)
        response.headers['Content-Disposition'] = f'attachment; filename={city}_historical_weather_data_{actual_days}days.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        print(f"Download error: {e}")
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
    """Download ALL historical weather data as CSV - NO FUTURE PREDICTIONS, NO DUPLICATES"""
    try:
        today = datetime.now().date()
        print(f"Download ALL historical data for {city} - up to {today}")

        # Get ONLY historical data (past dates including today), latest version only
        from sqlalchemy import func

        # Subquery to get the latest generation_timestamp for each PAST date
        subquery = db.session.query(
            Prediction.prediction_date,
            func.max(Prediction.generation_timestamp).label('max_timestamp')
        ).filter(
            Prediction.city == city,
            Prediction.prediction_date <= today  # ONLY DATES UP TO AND INCLUDING TODAY
        ).group_by(Prediction.prediction_date).subquery()

        # Get predictions with the latest timestamp for each PAST date
        predictions = Prediction.query.join(
            subquery,
            (Prediction.prediction_date == subquery.c.prediction_date) &
            (Prediction.generation_timestamp == subquery.c.max_timestamp)
        ).order_by(Prediction.prediction_date.asc()).all()

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No historical data found for this city'
            })

        print(f"Downloading {len(predictions)} historical records for {city} (up to {today})")

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Date', 'Min Temperature (°C)', 'Max Temperature (°C)',
            'Average Temperature (°C)', 'Humidity (%)',
            'Wind Speed (km/h)', 'Weather Condition', 'Model Version',
            'Prediction Generated At', 'Version', 'Is Current'
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
                pred.version,
                'Yes' if pred.is_current else 'No'
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename={city}_all_historical_weather_data.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        print(f"Download error: {e}")
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

@app.route('/fix-column-length')
def fix_column_length():
    try:
        from sqlalchemy import text
        db.session.execute(text("ALTER TABLE predictions ALTER COLUMN condition TYPE TEXT;"))
        db.session.commit()
        return "Success , condition column now supports unlimited length. All predictions will save properly."
    except Exception as e:
        return f"Error: {str(e)}"






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

@app.route('/generate-daily-predictions')
def generate_daily_predictions():
    """Generate predictions for multiple cities automatically , for cron jobs"""
    try:
        # List of cities you want daily predictions for
        cities = ["Amman"]
        
        results = []
        
        for city in cities:
            try:
                print(f"Generating predictions for {city}...")
                
                # Train model for this city
                training_result = predictor.train_model(city)
                if 'error' in training_result:
                    results.append({
                        "city": city,
                        "status": "error", 
                        "message": f"Training failed: {training_result['error']}"
                    })
                    print(f"Training failed for {city}: {training_result['error']}")
                    continue
                
                # Generate predictions
                prediction_result = predictor.predict_weather(days=7)
                if 'error' in prediction_result:
                    results.append({
                        "city": city,
                        "status": "error",
                        "message": f"Prediction failed: {prediction_result['error']}"
                    })
                    print(f"Prediction failed for {city}: {prediction_result['error']}")
                    continue
                
                # Get model metrics
                model_metrics = {}
                model_name = "Unknown"
                
                if hasattr(predictor, 'results') and predictor.best_model_name:
                    best_model_result = predictor.results[predictor.best_model_name]
                    model_metrics = best_model_result.get('detailed_metrics', {})
                    model_name = predictor.best_model_name
                
                # Save to database
                save_success = save_predictions_to_db(
                    city,
                    prediction_result['predictions'],
                    model_name,
                    model_metrics
                )
                
                if save_success:
                    results.append({
                        "city": city,
                        "status": "success",
                        "message": f"Generated {len(prediction_result['predictions'])} predictions using {model_name}",
                        "model_used": model_name,
                        "predictions_generated": len(prediction_result['predictions'])
                    })
                    print(f"Successfully generated and saved predictions for {city} using {model_name}")
                else:
                    results.append({
                        "city": city,
                        "status": "error",
                        "message": "Failed to save predictions to database"
                    })
                    print(f"Database save failed for {city}")
                    
            except Exception as e:
                results.append({
                    "city": city,
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}"
                })
                print(f"Unexpected error for {city}: {e}")
        
        # Clean up old predictions
        cleanup_old_predictions()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "total_cities_processed": len(cities),
            "successful": len([r for r in results if r['status'] == 'success']),
            "failed": len([r for r in results if r['status'] == 'error']),
            "results": results
        }
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Global error: {str(e)}"
        }), 500
















@app.route('/migrate-database', methods=['GET', 'POST'])
def migrate_database():
    """Add model_comparison column to existing database"""
    try:
        with app.app_context():
            # Check if model_comparison column exists
            db_path = os.path.join('instance', 'weather_predictions.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(model_performance)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'model_comparison' not in columns:
                cursor.execute('ALTER TABLE model_performance ADD COLUMN model_comparison JSON')
                print("✓ Added model_comparison column to model_performance table")
                
                # Update existing records with empty model_comparison
                cursor.execute('UPDATE model_performance SET model_comparison = ?', (json.dumps({}),))
                print("✓ Updated existing records with empty model_comparison")
                
                message = "Database migration completed successfully!"
            else:
                message = "Database already has the model_comparison column."
            
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': message})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})




@app.route('/fix-database', methods=['GET', 'POST'])
def fix_database():
    """Fix PostgreSQL database schema for deployed version"""
    try:
        with app.app_context():
            from sqlalchemy import text
            
            # Check if we're using PostgreSQL
            if 'postgresql' in app.config["SQLALCHEMY_DATABASE_URI"]:
                try:
                    # Check if model_comparison column exists
                    check_query = text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name='model_performance' AND column_name='model_comparison'
                    """)
                    
                    result = db.session.execute(check_query).fetchone()
                    
                    if not result:
                        # Add the column to PostgreSQL
                        alter_query = text("ALTER TABLE model_performance ADD COLUMN model_comparison JSONB")
                        db.session.execute(alter_query)
                        
                        # Set default empty JSON for existing records
                        update_query = text("UPDATE model_performance SET model_comparison = '{}'::jsonb")
                        db.session.execute(update_query)
                        
                        db.session.commit()
                        message = "Successfully added model_comparison column to PostgreSQL database!"
                    else:
                        message = "model_comparison column already exists in PostgreSQL database!"
                        
                except Exception as e:
                    db.session.rollback()
                    message = f"Error: {str(e)}"
                    
            else:
                message = "Not a PostgreSQL database - no migration needed"
            
            return f"""
            <html>
                <head>
                    <title>Database Fix</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 40px; background: linear-gradient(to bottom, #1a2980, #26d0ce); color: white; }}
                        .container {{ max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; color: #333; }}
                        .success {{ color: #28a745; font-weight: bold; }}
                        .error {{ color: #dc3545; font-weight: bold; }}
                        .info {{ color: #17a2b8; }}
                        a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Database Migration</h1>
                        <p>{message}</p>
                        <p><a href="/">Return to SkySense</a></p>
                    </div>
                </body>
            </html>
            """
            
    except Exception as e:
        return f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 40px; background: linear-gradient(to bottom, #1a2980, #26d0ce); color: white;">
                <div style="max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; color: #333;">
                    <h1>Database Migration Error</h1>
                    <p style="color: #dc3545; font-weight: bold;">Error: {str(e)}</p>
                    <p><a href="/" style="color: #667eea; text-decoration: none; font-weight: bold;">Return to SkySense</a></p>
                </div>
            </body>
        </html>
        """


@app.route('/current-weather/<city>', methods=['GET'])
def get_current_weather(city):
    """Get current weather data for real-time dashboard"""
    try:
        url = f"{BASE_URL}/current.json"
        params = {
            "key": API_KEY,
            "q": city
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': 'Unable to fetch current weather data'
            })
        
        data = response.json()
        current = data['current']
        
        current_weather = {
            'temp_c': current['temp_c'],
            'humidity': current['humidity'],
            'wind_kph': current['wind_kph'],
            'condition': current['condition']['text'],
            'feelslike_c': current['feelslike_c'],
            'pressure_mb': current['pressure_mb'],
            'precip_mm': current['precip_mm'],
            'cloud': current['cloud'],
            'vis_km': current['vis_km'],
            'uv': current['uv']
        }
        
        return jsonify({
            'success': True,
            'city': city,
            'current_weather': current_weather
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/fix-current-flags')
def fix_current_flags():
    """One-time fix for is_current flags"""
    try:
        today = datetime.now().date()
        
        # Mark past predictions as not current
        past_updated = Prediction.query.filter(
            Prediction.prediction_date < today
        ).update({'is_current': False})
        
        # Mark current/future as current  
        future_updated = Prediction.query.filter(
            Prediction.prediction_date >= today
        ).update({'is_current': True})
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Fixed {past_updated} past records and {future_updated} current/future records',
            'past_updated': past_updated,
            'future_updated': future_updated
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/trigger-daily-predictions')
def trigger_daily_predictions():
    """route for UptimeRobot to trigger daily predictions"""
    try:
        # Just call the existing route internally
        with app.test_client() as client:
            response = client.get('/generate-daily-predictions')
            
        return jsonify({
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'message': 'Daily predictions triggered',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created/verified")
        print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    app.run(debug=True)
