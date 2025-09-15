from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
model = joblib.load('rf_classifier.joblib')
pipeline = joblib.load('preprocessing_pipeline.joblib')


def preprocess_data(data):
    # Convert form data to DataFrame
    data = pd.DataFrame([data])
    # print("Initial Data:\n", data)
    
    # Convert categorical variables to numeric
    data['Chest Pain Type'] = data['Chest Pain Type'].replace({
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3,
        'Typical Angina': 0
    })
    data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0})
    data['Smoking'] = data['Smoking'].replace({
        'Current': 1,
        'Never': 0,
        'Former': 2
    })
    data['Alcohol Intake'] = data['Alcohol Intake'].replace({
        'Heavy': 1,
        'NaN': np.nan,
        'Moderate': 2
    })
    data['Family History'] = data['Family History'].replace({'Yes': 1, 'No': 0})
    data['Diabetes'] = data['Diabetes'].replace({'Yes': 1, 'No': 0})
    data['Obesity'] = data['Obesity'].replace({'Yes': 1, 'No': 0})
    data['Exercise Induced Angina'] = data['Exercise Induced Angina'].replace({'Yes': 1, 'No': 0})

    # print("After converting categorical variables:\n", data)

    # Feature engineering
    data['Log Blood Pressure'] = np.log(data['Blood Pressure'] + 1)
    data['Cholesterol_BloodPressure'] = data['Cholesterol'] * data['Blood Pressure']
    data['Exercise_Stress'] = data['Exercise Hours'] * data['Stress Level']
    data['Cholesterol_Ratio'] = data['Cholesterol'] / (data['Cholesterol'] + 1)
    data['Mean Arterial Pressure'] = data['Blood Pressure'] + (data['Blood Pressure'] - data['Blood Pressure']) / 3
    data['Risk_Score'] = (0.3 * data['Age'] +
                          0.2 * data['Cholesterol_Ratio'] +
                          0.2 * data['Log Blood Pressure'] +
                          0.1 * data['Obesity'] +
                          0.1 * data['Stress Level'] +
                          0.1 * data['Smoking'])

    # print("After feature engineering:\n", data)

    # Selecting relevant features
    features = ['Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Smoking',
                'Alcohol Intake', 'Exercise Hours', 'Family History', 'Diabetes', 'Obesity',
                'Stress Level', 'Blood Sugar', 'Exercise Induced Angina', 'Chest Pain Type',
                'Log Blood Pressure', 'Cholesterol_BloodPressure', 'Exercise_Stress', 
                'Cholesterol_Ratio', 'Mean Arterial Pressure', 'Risk_Score']

    data = data[features]
    # print("After selecting relevant features:\n", data)
    data = np.array(data)
    # print("After converting to numpy array:\n", data)

    # Standardization and normalization
    data = pipeline.transform(data)
    # print("After standardization and normalization:\n", data)

    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    form_data = {key: float(value) if key not in ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type'] else value for key, value in form_data.items()}

    processed_features = preprocess_data(form_data)
    # print(processed_features)
    prediction = model.predict(processed_features)
    
    output = prediction[0]
    # print(output)
    return render_template('result.html', prediction_text=f'Heart Disease Prediction: {"Yes" if output == 1 else "No"}')

if __name__ == "__main__":
    app.run(debug=True)
