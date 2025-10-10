from . import db
from datetime import datetime

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
