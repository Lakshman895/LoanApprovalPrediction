import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_data, y_train_data, X_test_data, y_test_data):
        try:
            logging.info('Split training and testing data')

            X_train, y_train, X_test, y_test = (X_train_data, y_train_data, X_test_data, y_test_data)

            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'KNeighbors Classifier': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }

            params = {
                'Logistic Regression': {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear']
                },
                'KNeighbors Classifier': {
                    'n_neighbors': list(range(3, 20)),
                    'weights': ['uniform', 'distance']
                },
                'Decision Tree': {
                    'max_depth': list(range(1, 15)),
                    'criterion': ['gini', 'entropy']
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': list(range(5, 15)),
                    'criterion': ['gini', 'entropy']
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': list(range(3, 10))
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': list(range(3, 10)),
                    'gamma': [0, 0.1, 0.5, 1]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly']
                }
            }

            logging.info('Getting the report from the utils file')

            model_report: pd.DataFrame = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            logging.info('Getting model evaluation report')

            def score_model(row):
                return (
                    row['Test_Accuracy'] * 0.4 +
                    row['Train_Accuracy'] * 0.3 +
                    (1 - row['Log_Loss']) * 0.2 +
                    row['Cross_val_score'] * 0.1
                )

            model_report['Score'] = model_report.apply(score_model, axis=1)

            logging.info(f'Model report:\n{model_report}')

            best_model_name = model_report.loc[model_report['Score'].idxmax(), 'Model']
            best_model_score = model_report['Score'].max()

            best_model = models[best_model_name]

            logging.info(f'Best model {best_model_name} with score {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            return accuracy

        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e, sys)

if __name__ == '__main__':
    print('Everything is OK')

