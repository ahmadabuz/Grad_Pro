# __init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

def create_app():
    app = Flask(__name__)
    
    # Database configuration
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///weather_predictions.db"
    app.config["SQLALCHEMY_ECHO"] = False
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize db with app
    db.init_app(app)
    
    # Import and register models
    with app.app_context():
        from . import models
        
    # Import and register routes
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app
