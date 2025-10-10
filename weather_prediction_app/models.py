from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    prediction_date = db.Column(db.Date, nullable=False)  # The date being predicted
    generation_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    model_version = db.Column(db.String(50), nullable=False)

    # Prediction values
    min_temp = db.Column(db.Float, nullable=False)
    max_temp = db.Column(db.Float, nullable=False)
    avg_temp = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    condition = db.Column(db.String(100), nullable=False)

    #for historical tracking
    is_current = db.Column(db.Boolean, default=True)  # Mark most recent prediction
    version = db.Column(db.Integer, default=1)  # Version number

    # Index for faster queries (speed boosters)
    __table_args__ = (
        db.Index('idx_city_date', 'city', 'prediction_date'),
        db.Index('idx_generation_timestamp', 'generation_timestamp'),
    )

class ModelPerformance(db.Model):
    __tablename__ = "model_performance"

    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    model_name = db.Column(db.String(50), nullable=False)
    r2_score = db.Column(db.Float, nullable=False)
    mae = db.Column(db.Float, nullable=False)
    rmse = db.Column(db.Float, nullable=False)

    # Detailed metrics (stored as JSON)
    detailed_metrics = db.Column(db.JSON, nullable=False)
