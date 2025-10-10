# routes.py
from flask import Blueprint, render_template, request, jsonify, send_file
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import warnings
import time
import json
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import threading
import csv
from io import StringIO
from flask import make_response

from . import db
from .models import Prediction, ModelPerformance

bp = Blueprint('main', __name__)

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
np.random.seed(42)

# Configuration
API_KEY = "245159b18d634837900112029250310"
BASE_URL = "https://api.weatherapi.com/v1"

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

    # [PASTE YOUR ENTIRE WeatherPredictor CLASS HERE - ALL METHODS]
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
        # [PASTE YOUR COMPLETE fetch_weather_data METHOD]
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

    # [CONTINUE PASTING ALL YOUR WeatherPredictor METHODS...]
    def prepare_advanced_features(self, data):
        # [PASTE YOUR COMPLETE prepare_advanced_features METHOD]
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
        # [PASTE YOUR COMPLETE train_advanced_models METHOD]
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
            # Use MultiOutputRegressor for linear models and tree-based models that don't natively support multi-output
            if name in ['ElasticNet', 'Ridge', 'Lasso', 'GradientBoosting', 'XGBoost', 'LightGBM']:
                multi_model = MultiOutputRegressor(model)
                multi_model.fit(X_train, y_train)
                y_pred = multi_model.predict(X_test)
                trained_model = multi_model
            else:
                # RandomForestRegressor can handle multi-output natively
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                trained_model = model

            # Calculate metrics for each target
            metrics = {}
            for i, target_name in enumerate(['avg_c', 'min_c', 'max_c', 'humidity', 'wind_kph']):
                metrics[target_name] = {
                    'mae': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                    'rmse': np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])),
                    'r2': r2_score(y_test[:, i], y_pred[:, i])
                }

            # Overall metrics (average across targets)
            overall_metrics = {
                'mae': np.mean([m['mae'] for m in metrics.values()]),
                'rmse': np.mean([m['rmse'] for m in metrics.values()]),
                'r2': np.mean([m['r2'] for m in metrics.values()]),
                'model': trained_model,
                'detailed_metrics': metrics
            }

            results[name] = overall_metrics

        return results

    # [CONTINUE PASTING ALL REMAINING WeatherPredictor METHODS...]
    # create_future_features, train_model, predict_weather, determine_weather_condition, generate_chart

# Initialize predictor
predictor = WeatherPredictor()

# [PASTE ALL YOUR HELPER FUNCTIONS]
def ensure_instance_folder():
    instance_path = os.path.join(os.path.dirname(__file__), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        print(f"Created instance folder at: {instance_path}")
    return instance_path

def init_db(app):
    with app.app_context():
        try:
            ensure_instance_folder()
            db.create_all()
            print("✓ Database tables created successfully")
        except Exception as e:
            print(f"Database initialization error: {e}")

def save_predictions_to_db(city, predictions, model_name, model_metrics):
    # [PASTE YOUR COMPLETE save_predictions_to_db FUNCTION]
    try:
        generation_time = datetime.now()
        today = datetime.now().date()

        # First, mark all previous predictions for this city as not current
        Prediction.query.filter(
            Prediction.city == city,
            Prediction.prediction_date >= today,
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

            # Only save predictions for today and future dates
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

        # Calculate overall R² score from detailed metrics
        overall_r2 = 0
        if model_metrics and isinstance(model_metrics, dict):
            # Extract R² scores from each parameter and average them
            r2_scores = []
            for param_metrics in model_metrics.values():
                if isinstance(param_metrics, dict) and 'r2' in param_metrics:
                    r2_scores.append(param_metrics['r2'])
            if r2_scores:
                overall_r2 = sum(r2_scores) / len(r2_scores)

        # Always save performance metrics
        # Calculate overall MAE and RMSE from detailed metrics
        overall_mae = 0
        overall_rmse = 0
        if model_metrics and isinstance(model_metrics, dict):
            # Extract MAE and RMSE scores from each parameter and average them
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

        # Always save performance metrics
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
    # [PASTE YOUR COMPLETE get_latest_predictions_from_db FUNCTION]
    try:
        today = datetime.now().date()

        # Get the most recent CURRENT predictions for this city starting from today
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

# [PASTE ALL YOUR ROUTES]
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    # [PASTE YOUR COMPLETE predict ROUTE]
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

# [PASTE ALL YOUR OTHER ROUTES...]
@bp.route('/cities', methods=['GET'])
def get_cities():
    # [PASTE YOUR get_cities ROUTE]
    query = request.args.get('q', '')  # what the user typed
    if not query:
        return jsonify([])

    try:
        url = f"{BASE_URL}/search.json"
        params = {"key": API_KEY, "q": query}
        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            return jsonify([])

        cities = response.json()
        # Extract name + country for dropdown
        results = [
            {"name": f"{c['name']}, {c['country']}", "value": c['name']}
            for c in cities
        ]
        return jsonify(results)

    except Exception as e:
        print("Error fetching cities:", e)
        return jsonify([])

# [CONTINUE WITH ALL YOUR OTHER ROUTES...]

# Background task to clear old cache entries daily
def cache_cleanup_task():
    while True:
        time.sleep(86400)  # Sleep for 24 hours
        predictor.clear_old_cache()
        print("Cleared old cache entries")

# Start the background cleanup thread
cache_thread = threading.Thread(target=cache_cleanup_task, daemon=True)
cache_thread.start()
