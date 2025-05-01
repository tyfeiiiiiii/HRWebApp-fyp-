# from flask import Flask, jsonify, request
# import joblib
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model
# model1 = joblib.load('models/random_forest_model_set1.pkl')
# model2 = joblib.load('models/random_forest_model_set2.pkl')

# # Define the expected column names (must match the model training)
# columns1=[
#     'OverTime',
#     'MaritalStatus', 
#     'MonthlyIncome', 
#     'StockOptionLevel',
#     'BusinessTravel', 
#     'TotalWorkingYears', 
#     'JobInvolvement',
#     'YearsAtCompany', 
#     'Age', 
#     'DistanceFromHome'
# ]

# columns2 = [
#     'satisfaction_level',
#     'last_evaluation',
#     'number_project',
#     'average_montly_hours',  # Note: typo in "montly" must match model
#     'time_spend_company',
#     'Work_accident',          # Case-sensitive!
#     'promotion_last_5years',
#     'department',
#     'salary'
# ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract form data
#         data1 = {
#             'OverTime': request.form['OverTime'],  # Assuming OverTime is categorical (e.g., Yes/No), no need for float conversion
#             'MaritalStatus': request.form['MaritalStatus'],  # Assuming this is a categorical field (e.g., Married/Single)
#             'MonthlyIncome': float(request.form['MonthlyIncome']),  # Should be a float as it's a monetary value
#             'StockOptionLevel': int(request.form['StockOptionLevel']),  # This seems to be a categorical variable represented as integer
#             'BusinessTravel': request.form['BusinessTravel'],  # This is categorical (e.g., 'Travel_Rarely', 'Travel_Frequently')
#             'TotalWorkingYears': int(request.form['TotalWorkingYears']),  # Integer as it's the number of years
#             'JobInvolvement': int(request.form['JobInvolvement']),  # This seems like a categorical variable, so int is used
#             'YearsAtCompany': int(request.form['YearsAtCompany']),  # Integer as it's the number of years
#             'Age': int(request.form['Age']),  # Age is integer
#             'DistanceFromHome': int(request.form['DistanceFromHome']),  # Distance is an integer value
#         }

#         data2 = {
#             'satisfaction_level': float(request.form['satisfactionLevel']),
#             'last_evaluation': float(request.form['lastEvaluation']),
#             'number_project': int(request.form['numberProject']),
#             'average_montly_hours': int(request.form['averageMonthlyHours']),
#             'time_spend_company': int(request.form['timeSpendCompany']),
#             'Work_accident': int(request.form['workAccident']),
#             'promotion_last_5years': int(request.form['promotionLast5Years']),
#             'department': request.form['department'],
#             'salary': request.form['salary']
#         }

#         # Convert to DataFrame with correct column order
#         df1 = pd.DataFrame([data1], columns=columns1)
#         df2 = pd.DataFrame([data2], columns=columns2)

#         # Make prediction
#         prediction_set1= model1.predict(df1)
#         prediction_set2 = model2.predict(df2)

#         return jsonify({
#             'prediction_model1': int(prediction_set1[0]),
#             'prediction_model2': int(prediction_set2[0])
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, jsonify, request
import joblib
import pandas as pd
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the trained models
model1 = joblib.load('models/random_forest_model_set1.pkl')
model2 = joblib.load('models/random_forest_model_set2.pkl')

# Define the expected column names
columns1 = [
    'OverTime', 'MaritalStatus', 'MonthlyIncome', 'StockOptionLevel',
    'BusinessTravel', 'TotalWorkingYears', 'JobInvolvement', 'YearsAtCompany',
    'Age', 'DistanceFromHome'
]

columns2 = [
    'satisfaction_level', 'last_evaluation', 'average_montly_hours',
    'Work_accident', 'promotion_last_5years', 'department'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the JSON data sent from the client

        # Prepare data for prediction
        data1 = {
            'OverTime': data['OverTime'],
            'MaritalStatus': data['MaritalStatus'],
            'MonthlyIncome': float(data['MonthlyIncome']),
            'StockOptionLevel': int(data['StockOptionLevel']),
            'BusinessTravel': data['BusinessTravel'],
            'TotalWorkingYears': int(data['TotalWorkingYears']),
            'JobInvolvement': int(data['JobInvolvement']),
            'YearsAtCompany': int(data['YearsAtCompany']),
            'Age': int(data['Age']),
            'DistanceFromHome': int(data['DistanceFromHome']),
        }

        data2 = {
            'satisfaction_level': float(data['SatisfactionLevel']),
            'last_evaluation': float(data['LastEvaluation']),
            'average_montly_hours': int(data['AverageMonthlyHours']),
            'Work_accident': int(data['WorkAccident']),
            'promotion_last_5years': int(data['PromotionLast5Years']),
            'department': data['Department'],
        }

        # Convert to DataFrame with correct column order
        df1 = pd.DataFrame([data1], columns=columns1)
        df2 = pd.DataFrame([data2], columns=columns2)

        # Make prediction
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
