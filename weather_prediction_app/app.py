# app.py
from weather_prediction_app import create_app, init_db

app = create_app()

# Initialize database
init_db(app)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("Starting Weather Prediction Server:")
    print("Cache system enabled - first request will train, subsequent requests will be instant")
    app.run(debug=True, host='127.0.0.1', port=5000)
