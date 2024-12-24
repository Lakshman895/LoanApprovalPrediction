from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pickle

# Initialize Flask application
application = Flask(__name__)
app = application

# File paths
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

# Check and load model and preprocessor during startup
model = None
preprocessor = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")

    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
        print("Preprocessor loaded successfully.")
    else:
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            # Collect and preprocess input data
            data = CustomData(
                Gender=request.form.get('Gender').lower(),
                Married=request.form.get('Married').lower(),
                Dependents=int(request.form.get('Dependents')),
                Education=request.form.get('Education').lower(),
                Self_Employed=request.form.get('Self_Employed').lower(),
                LoanAmount=float(request.form.get('LoanAmount')),
                Loan_Amount_Term=int(request.form.get('Loan_Amount_Term')),
                Credit_History=float(request.form.get('Credit_History')),
                Property_Area=request.form.get('Property_Area').lower(),
                ApplicantIncome=int(request.form.get('ApplicantIncome')),
                CoapplicantIncome=int(request.form.get('CoapplicantIncome')),
            )

            pred_df = data.get_data_as_data_frame()
            print('The data to be predicted:\n', pred_df)

            # Make prediction
            predict_pipeline = PredictPipeline()
            print('Predict pipeline initialized.')
            log_results = predict_pipeline.predict(pred_df)
            print('Log result:', log_results)
            results = np.expm1(log_results)  # Convert log results to actual values
            print('The predicted result is:', results)

            return render_template('home.html', results=results[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction.")

if __name__ == "__main__":
    # Run the application
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
