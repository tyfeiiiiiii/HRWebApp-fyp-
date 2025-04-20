from flask import Flask, jsonify, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/random_forest_model_set2.pkl')

# Define the expected column names (must match the model training)
columns = [
    'satisfaction_level',
    'last_evaluation',
    'number_project',
    'average_montly_hours',  # Note: typo in "montly" must match model
    'time_spend_company',
    'Work_accident',          # Case-sensitive!
    'promotion_last_5years',
    'department',
    'salary'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {
            'satisfaction_level': float(request.form['satisfactionLevel']),
            'last_evaluation': float(request.form['lastEvaluation']),
            'number_project': int(request.form['numberProject']),
            'average_montly_hours': int(request.form['averageMonthlyHours']),
            'time_spend_company': int(request.form['timeSpendCompany']),
            'Work_accident': int(request.form['workAccident']),
            'promotion_last_5years': int(request.form['promotionLast5Years']),
            'department': request.form['department'],
            'salary': request.form['salary']
        }

        # Convert to DataFrame with correct column order
        df = pd.DataFrame([data], columns=columns)

        # Make prediction
        prediction = model.predict(df)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
