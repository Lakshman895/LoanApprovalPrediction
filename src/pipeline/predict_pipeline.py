import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        print("Before Loading")
        
        # Loading the model and preprocessor
        try:
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
        except Exception as e:
            raise CustomException(f"Error loading model or preprocessor: {e}", sys)

        print("After Loading")
        
        # Transforming the features using preprocessor
        data_scaled = preprocessor.transform(features)
        print("The scaled data is: ", data_scaled.T)
        
        # Making continuous predictions (probabilities or scores)
        preds = model.predict(data_scaled)
        print("Continuous prediction output: ", preds)
        
        # Apply a threshold to convert the continuous value to binary
        threshold = 0.5  # Example threshold for binary classification
        binary_preds = [1 if pred >= threshold else 0 for pred in preds]
        
        return binary_preds  # Return binary classification results (0 or 1)

class CustomData:
    def __init__(self,
                 Gender: str,
                 Married: str,
                 Dependents: int,
                 Education: str,
                 Self_Employed: str,
                 LoanAmount: float,
                 Loan_Amount_Term: int,
                 Credit_History: float,
                 Property_Area: str,
                 ApplicantIncome: int,
                 CoapplicantIncome:int):
        
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Gender': [self.Gender],
                'Married': [self.Married],
                'Dependents': [self.Dependents],
                'Education': [self.Education],
                'Self_Employed': [self.Self_Employed],
                'LoanAmount': [self.LoanAmount],
                'Loan_Amount_Term': [self.Loan_Amount_Term],
                'Credit_History': [self.Credit_History],
                'Property_Area': [self.Property_Area],
                'ApplicantIncome': [self.ApplicantIncome],
                'CoapplicantIncome' : [self.CoapplicantIncome]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
