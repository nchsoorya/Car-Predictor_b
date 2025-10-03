import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for the frontend to access this API
CORS(app)

# --- 1. Model Loading and Configuration ---
MODEL_FILE = 'xgb_car_price_model.joblib'
MODEL_LOADED = False
model = None

# Features must be in the exact order/name used during training
# These names match the inputs from your HTML form
EXPECTED_FEATURES = [
    'vehicle_age', 'km_driven', 'brand', 'seller_type', 'fuel_type',
    'transmission_type', 'mileage', 'engine', 'max_power', 'seats',
    'accident_history', 'num_claims', 'total_claim_amount'
]

# Load the entire pipeline (preprocessor + model)
try:
    model = joblib.load(MODEL_FILE)
    MODEL_LOADED = True
    print(f"Successfully loaded model: {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}. Ensure {MODEL_FILE} is in the same directory.")

# --- 2. Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model failed to load on server."}), 500

    try:
        # Get JSON data from the request (sent by index.html)
        data = request.get_json(force=True)

        # Convert the single input dictionary into a pandas DataFrame (required by the pipeline)
        # We ensure the columns are in the correct order for the pipeline
        input_df = pd.DataFrame([data], columns=EXPECTED_FEATURES)
        
        # 3. Make the prediction (output is on the log-scale)
        prediction_log = model.predict(input_df)[0]
        
        # 4. Inverse transform the result: exp(x) - 1 to get the price in Rupees
        price = np.expm1(prediction_log)

        # Return the final predicted price as a JSON object
        return jsonify({'predicted_price': round(price)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Invalid input or prediction failed. Error: {str(e)}"}), 400

if __name__ == '__main__':
    # For local testing, run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
