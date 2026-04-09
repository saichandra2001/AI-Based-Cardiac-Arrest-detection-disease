from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))  # IMPORTANT
    print("✅ Model & Scaler loaded successfully!")
except Exception as e:
    model = None
    scaler = None
    print(f"❌ Error loading files: {e}")

# Feature names (VERY IMPORTANT)
feature_names = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "❌ Model not loaded properly"

    try:
        # Get input
        features = [float(x) for x in request.form.values()]

        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        result = "🔴 High Risk of Heart Disease" if prediction == 1 else "🟢 Low Risk (Healthy)"

        return render_template('index.html', prediction_text=f'Result: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text="⚠️ Error in prediction")

if __name__ == "__main__":
    pp.run(host="0.0.0.0", port=10000)