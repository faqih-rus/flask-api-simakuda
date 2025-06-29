from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import os
import time
from datetime import datetime
import traceback

app = Flask(__name__)

# Global variables
model = None
scaler = None
label_encoder = None
time_features_scaler = None
time_window = 5
forecast_horizon = 10
model_type = "unknown"  # Will store model type (single input or dual input)

def load_model_and_preprocessors():
    """
    Load models and detect model type
    """
    global model, scaler, label_encoder, time_features_scaler, time_window, forecast_horizon, model_type
    
    print("Loading air quality forecasting model...")
    model_path = os.path.join('models', 'saved', 'air_quality_60k_model.h5')
    scaler_path = os.path.join('models', 'saved', 'scaler_60k.pkl')
    encoder_path = os.path.join('models', 'saved', 'label_encoder_60k.pkl')
    config_path = os.path.join('models', 'saved', 'time_config.pkl')
    time_scaler_path = os.path.join('models', 'saved', 'time_scaler_60k.pkl')

    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Determine model type (single input or dual input)
        if isinstance(model.input, list) and len(model.input) == 2:
            model_type = "dual_input"
            print("Detected dual-input model (gas + time features)")
        else:
            model_type = "single_input"
            print("Detected single-input model (gas only)")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully")
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully")
        print(f"Classes: {label_encoder.classes_}")
        
        # Try to load time features scaler
        if os.path.exists(time_scaler_path):
            with open(time_scaler_path, 'rb') as f:
                time_features_scaler = pickle.load(f)
            print("Time features scaler loaded successfully")
        else:
            print("Time features scaler not found, using default normalization")
        
        # Try to load time config
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                time_window = config['time_window']
                forecast_horizon = config['forecast_horizon']
            print(f"Using time window: {time_window}, forecast horizon: {forecast_horizon} seconds")
        
        return True
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        print("API will start but predictions won't work until model is available")
        return False

# Load model on startup
model_loaded = load_model_and_preprocessors()

@app.route('/')
def home():
    """
    Home page - shows the web interface if model is loaded
    """
    return render_template('index.html', model_ready=model_loaded)

@app.route('/api', methods=['GET'])
def api_home():
    """
    API information endpoint
    """
    return jsonify({
        'status': 'success',
        'message': 'Air Quality Prediction API is running',
        'model': 'Neural Network 2-16-8-3 (sesuai skripsi)',
        'endpoints': {
            '/predict': 'POST request with CO and CO2 values'
        },
        'model_loaded': model is not None
    })

def create_gas_sequence(co_ppm, co2_ppm):
    """
    Create a realistic gas reading sequence leading to current values
    """
    # Create a trend with small random variations
    co_base = co_ppm * 0.95
    co2_base = co2_ppm * 0.97
    
    # Generate sequences with small trend
    co_seq = np.linspace(co_base, co_ppm, time_window)
    co2_seq = np.linspace(co2_base, co2_ppm, time_window)
    
    # Add small random variations
    co_seq += np.random.normal(0, co_ppm * 0.02, time_window)
    co2_seq += np.random.normal(0, co2_ppm * 0.01, time_window)
    
    # Ensure values are positive
    co_seq = np.maximum(co_seq, 0)
    co2_seq = np.maximum(co2_seq, 0)
    
    # Combine into sequence
    gas_sequence = np.column_stack([co_seq, co2_seq])
    
    # Scale sequence
    gas_sequence_scaled = scaler.transform(gas_sequence)
    
    # Add batch dimension
    gas_sequence_scaled = np.expand_dims(gas_sequence_scaled, axis=0)
    
    return gas_sequence_scaled

def create_time_sequence(hour, minute, day_of_week):
    """
    Create time feature sequence for a given time
    """
    # Create time features for a single timepoint
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Combine into single time feature vector
    time_features = []
    
    # Create sequence of timepoints (advancing slightly for each step)
    time_sequence = np.zeros((time_window, 6))
    
    for i in range(time_window):
        # Adjust time slightly for each step in sequence
        # This simulates time advancing in the sequence
        adj_minute = (minute + i/time_window) % 60
        adj_hour = (hour + (minute + i/time_window) // 60) % 24
        
        # Calculate features for this timepoint
        time_sequence[i, 0] = np.sin(2 * np.pi * adj_hour / 24)
        time_sequence[i, 1] = np.cos(2 * np.pi * adj_hour / 24)
        time_sequence[i, 2] = np.sin(2 * np.pi * adj_minute / 60)
        time_sequence[i, 3] = np.cos(2 * np.pi * adj_minute / 60)
        time_sequence[i, 4] = day_sin
        time_sequence[i, 5] = day_cos
    
    # Scale if needed
    if time_features_scaler is not None:
        time_sequence = time_features_scaler.transform(time_sequence)
    
    # Add batch dimension
    time_sequence = np.expand_dims(time_sequence, axis=0)
    
    return time_sequence

def predict_with_uncertainty(gas_sequence, time_sequence=None, num_samples=3):
    """
    Make time-aware predictions with uncertainty modeling
    Handles both single-input and dual-input models
    """
    global model_type
    
    # Run multiple forward passes for uncertainty
    predictions = []
    
    try:
        for _ in range(num_samples):
            if model_type == "dual_input":
                # For dual-input model (gas + time)
                # Use training=True to enable dropout for MC Dropout
                pred = model([gas_sequence, time_sequence], training=True)
            else:
                # For single-input model (gas only)
                pred = model(gas_sequence, training=True)
            
            predictions.append(pred.numpy()[0])
        
        # Average the predictions
        mean_prediction = np.mean(predictions, axis=0)
        
        # Add small random variations to simulate future uncertainty
        future_uncertainty = np.random.normal(0, 0.03 * forecast_horizon/10, size=mean_prediction.shape)
        
        # Apply uncertainty and renormalize
        prediction_with_uncertainty = mean_prediction + future_uncertainty
        prediction_with_uncertainty = np.clip(prediction_with_uncertainty, 0.01, 0.99)
        prediction_with_uncertainty = prediction_with_uncertainty / np.sum(prediction_with_uncertainty)
        
        return prediction_with_uncertainty
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        # Return a uniform distribution as a fallback
        return np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for 10-second ahead air quality prediction
    with time awareness
    """
    if not model_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please train the model first using train_model.py'
        }), 503
        
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Get gas reading parameters
        co_value = float(data.get('co', 0))
        co2_value = float(data.get('co2', 0))
        
        # Get time parameters (or use current time if not provided)
        now = datetime.now()
        hour = int(data.get('hour', now.hour))
        minute = int(data.get('minute', now.minute))
        day_of_week = int(data.get('day_of_week', now.weekday()))
        
        # Validate input
        if co_value < 0 or co2_value < 0:
            return jsonify({
                'status': 'error',
                'message': 'CO and CO2 values must be positive'
            }), 400
        
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= day_of_week < 7):
            return jsonify({
                'status': 'error',
                'message': 'Invalid time parameters'
            }), 400
        
        # Create gas sequence input
        # For a single gas reading, we create a sequence by repeating with small variations
        gas_sequence = np.array([[co_value, co2_value]] * time_window)
        gas_sequence_scaled = scaler.transform(gas_sequence)
        gas_sequence_scaled = np.expand_dims(gas_sequence_scaled, axis=0)
        
        # Create time input if needed
        if model_type == "dual_input":
            # Create cyclical time features
            time_sequence = np.zeros((time_window, 6))
            
            for i in range(time_window):
                # Calculate time features for this position in sequence
                adj_minute = (minute + i/time_window) % 60
                adj_hour = (hour + (minute + i/time_window) // 60) % 24
                
                # Cyclical encoding for time
                time_sequence[i, 0] = np.sin(2 * np.pi * adj_hour / 24)
                time_sequence[i, 1] = np.cos(2 * np.pi * adj_hour / 24)
                time_sequence[i, 2] = np.sin(2 * np.pi * adj_minute / 60)
                time_sequence[i, 3] = np.cos(2 * np.pi * adj_minute / 60)
                time_sequence[i, 4] = np.sin(2 * np.pi * day_of_week / 7)
                time_sequence[i, 5] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Scale time features if we have a scaler
            if time_features_scaler is not None:
                time_sequence = time_features_scaler.transform(time_sequence)
            
            # Add batch dimension
            time_sequence = np.expand_dims(time_sequence, axis=0)
        else:
            time_sequence = None
        
        # Make prediction with timing
        start_time = time.time()
        prediction = predict_with_uncertainty(gas_sequence_scaled, time_sequence)
        prediction_time = (time.time() - start_time) * 1000  # ms
        
        # Process results
        predicted_class_idx = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(prediction[predicted_class_idx] * 100)
        
        # Get all class probabilities
        class_probabilities = {
            class_name: float(prediction[i]) * 100 
            for i, class_name in enumerate(label_encoder.classes_)
        }
        
        # Return prediction with detailed forecast information
        return jsonify({
            'status': 'success',
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'input': {
                'co': co_value,
                'co2': co2_value,
                'time': f"{hour:02d}:{minute:02d}",
                'day': ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week]
            },
            'forecast': {
                'horizon_seconds': forecast_horizon,
                'description': f'Air quality prediction for exactly {forecast_horizon} seconds in the future',
                'time_aware': model_type == "dual_input"
            },
            'prediction_time_ms': prediction_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Return information about the loaded model
    """
    if not model_loaded:
        return jsonify({
            'status': 'not_loaded',
            'message': 'Model not loaded'
        }), 404
    
    try:
        # Get model architecture summary
        model_details = []
        model.summary(print_fn=lambda x: model_details.append(x))
        
        return jsonify({
            'status': 'success',
            'model_type': model_type,
            'time_window': time_window,
            'forecast_horizon': forecast_horizon,
            'classes': list(label_encoder.classes_),
            'architecture': model_details,
            'gas_features': ['CO', 'CO2'],
            'time_features': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos'] if model_type == "dual_input" else None,
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """
    Check if the model is loaded correctly
    """
    files_exist = (
        os.path.exists(model_path) and
        os.path.exists(scaler_path) and
        os.path.exists(encoder_path)
    )
    
    models_loaded = (model is not None and scaler is not None and label_encoder is not None)
    
    if files_exist and models_loaded:
        return jsonify({
            'status': 'healthy',
            'model_exists': True,
            'model_loaded': True,
            'classes': list(label_encoder.classes_) if label_encoder else None,
            'message': 'Model and preprocessors are loaded and ready'
        }), 200
    elif files_exist and not models_loaded:
        return jsonify({
            'status': 'warning',
            'model_exists': True,
            'model_loaded': False,
            'message': 'Model files exist but failed to load'
        }), 500
    else:
        return jsonify({
            'status': 'error',
            'model_exists': False,
            'model_loaded': False,
            'message': 'Model files not found. Run train_model.py first.'
        }), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)