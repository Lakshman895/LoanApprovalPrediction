import pickle
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
            
        logging.info('Dumped the object file into pickle file')
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report_data = []
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            param = params[model_name]
            
            logging.info(f'Performing the gridsearch: {model_name}')
            
            # Perform grid search with cross-validation
            gs = GridSearchCV(model, param, cv=5, scoring='accuracy')
            gs.fit(X_train, y_train)
            
            # Set best parameters and refit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            logging.info('Predicting the train and test data')
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Predict probabilities for Log Loss calculation
            y_pred_train_prob = model.predict_proba(X_train)[:, 1]  # Probability of positive class
            y_pred_test_prob = model.predict_proba(X_test)[:, 1]    # Probability of positive class
            
            # Calculate evaluation metrics
            train_model_score = accuracy_score(y_train, y_pred_train)
            test_model_score = accuracy_score(y_test, y_pred_test)
            cross_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Calculate Log Loss
            log_loss_train = log_loss(y_train, y_pred_train_prob)
            log_loss_test = log_loss(y_test, y_pred_test_prob)
            
            result = {
                'Model': model_name,
                'Train_Accuracy': train_model_score,
                'Test_Accuracy': test_model_score,
                'Cross_val_score': cross_score.mean(),
                'Log_Loss': log_loss_test  # Add Log Loss to the result
            }
            
            report_data.append(result)
            logging.info('Appending the scores for model: %s', model_name)
        
        # Create DataFrame from results and save to CSV
        report_df = pd.DataFrame(report_data)
        logging.info('Saving the model evaluation results into CSV file')
        report_df.to_csv('artifacts/model_result.csv', index=False)
        
        return report_df
        
    except Exception as e:
        logging.error(f'Error occurred during model evaluation: {e}')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
