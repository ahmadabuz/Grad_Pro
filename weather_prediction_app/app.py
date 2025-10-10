from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os

# Initialize SQLAlchemy without app context first
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    if 'RENDER' in os.environ:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/weather_predictions.db"
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///weather_predictions.db"
    
    app.config["SQLALCHEMY_ECHO"] = False
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize db with app
    db.init_app(app)
    
    # Import and register models
    from models import Prediction, ModelPerformance
    
    # Import and register routes
    from routes import init_routes
    init_routes(app, db)
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
