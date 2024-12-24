import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome','CoapplicantIncome',]

            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Entered initiate data transformation')

            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            # Encode categorical columns manually
            encode_columns = {
                'Gender': {'female': 0, 'male': 1},
                'Married': {'no': 0, 'yes': 1},
                'Education': {'no': 0, 'yes': 1},
                'Self_Employed': {'no': 0, 'yes': 1},
                'Property_Area': {'rural': 0, 'semiurban': 1, 'urban': 2},
                'Loan_Status': {'approved': 0, 'not approved': 1}
            }

            # Apply encoding for categorical columns
            for column, mapping in encode_columns.items():
                train_df[column] = train_df[column].map(mapping).astype('int')
                test_df[column] = test_df[column].map(mapping).astype('int')

            logging.info('Categorical columns encoding completed')

            # Separate target and features
            target_column_name = "Loan_Status"

            X_train_df = train_df.drop(columns=[target_column_name], axis=1)
            y_train_data = train_df[target_column_name]

            X_test_df = test_df.drop(columns=[target_column_name], axis=1)
            y_test_data = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply transformation on training and test data
            processed_train_data = preprocessing_obj.fit_transform(X_train_df)
            processed_test_data = preprocessing_obj.transform(X_test_df)

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return processed data and the target columns separately
            return (
                processed_train_data,  # Transformed training features
                y_train_data,          # Original target for training
                processed_test_data,   # Transformed test features
                y_test_data,           # Original target for testing
                self.data_transformation_config.preprocessor_obj_file_path  # Path of preprocessor object
            )

        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e, sys)
