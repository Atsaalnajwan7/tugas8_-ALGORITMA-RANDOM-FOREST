from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model dan scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model dan scaler berhasil dimuat")
except Exception as e:
    print("Error loading model/scaler:", e)
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cek model
        if model is None or scaler is None:
            return "Model atau scaler gagal dimuat"

        # Ambil input dari form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]

        # Convert ke numpy array
        input_data = np.array([features])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = {
            'prediction': int(prediction),
            'label': 'Berisiko Penyakit Jantung' if prediction == 1 else 'Tidak Berisiko',
            'confidence': round(max(probability) * 100, 2)
        }

        return render_template(
            'result.html',
            result=result,
            data=request.form
        )

    except Exception as e:
        return f"Error saat prediksi: {str(e)}"

# Railway / Gunicorn
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)