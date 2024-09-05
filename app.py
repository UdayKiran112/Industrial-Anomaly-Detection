from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Define the mse function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load your trained model and pass the custom objects
model = load_model('autoencoder_model.h5', custom_objects={'mse': mse})

# Load your scaler (if used during preprocessing)
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Optional: Create an index.html for the frontend

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (e.g., from a form or API call)
    data = request.json  # Or use request.form for form data

    # Preprocess the data (apply the same preprocessing as during training)
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    # Make prediction
    reconstruction = model.predict(data)
    reconstruction_error = np.mean(np.square(data - reconstruction), axis=1)

    # Define an anomaly threshold (based on training or business logic)
    threshold = 0.01  # Example threshold
    anomaly = reconstruction_error > threshold

    # Return the result
    return jsonify({'anomaly': bool(anomaly[0]), 'reconstruction_error': reconstruction_error[0]})

if __name__ == '__main__':
    app.run(debug=True)
