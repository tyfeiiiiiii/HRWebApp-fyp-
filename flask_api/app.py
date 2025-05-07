from flask import Flask, jsonify, request
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained models
model1 = joblib.load('models/logistic_regression_model_set1.pkl')
model2 = joblib.load('models/random_forest_pipeline_set2.pkl')

# Define expected columns
columns1 = [
    'OverTime', 'MaritalStatus', 'MonthlyIncome', 'StockOptionLevel',
    'BusinessTravel', 'TotalWorkingYears', 'JobInvolvement', 'YearsAtCompany',
    'Age', 'DistanceFromHome'
]

columns2 = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'Work_accident',
    'promotion_last_5years', 'department', 'salary'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Prepare data for model 1 (logistic regression)
        data1 = {
            "OverTime": data["OverTime"],
            "MaritalStatus": data["MaritalStatus"],
            "MonthlyIncome": data["MonthlyIncome"],
            "StockOptionLevel": data["StockOptionLevel"],
            "BusinessTravel": data["BusinessTravel"],
            "TotalWorkingYears": data["TotalWorkingYears"],
            "JobInvolvement": data["JobInvolvement"],
            "YearsAtCompany": data["YearsAtCompany"],
            "Age": data["Age"],
            "DistanceFromHome": data["DistanceFromHome"]
        }

        # Prepare data for model 2 (pipeline should handle preprocessing)
        data2 = {
            "satisfaction_level": data["SatisfactionLevel"],
            "last_evaluation": data["LastEvaluation"],
            "number_project": data["NumberProject"],
            "average_monthly_hours": data["AverageMonthlyHours"],
            "Work_accident": data["WorkAccident"],
            "promotion_last_5years": data["PromotionLast5Years"],
            "department": data["Department"],
            "salary": data["Salary"]
        }

        df1 = pd.DataFrame([data1], columns=columns1)
        df2 = pd.DataFrame([data2], columns=columns2)

        prediction_set1 = model1.predict(df1)
        prediction_set2 = model2.predict(df2)

        return jsonify({
            'prediction_model1': int(prediction_set1[0]),
            'prediction_model2': int(prediction_set2[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
