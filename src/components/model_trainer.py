import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X, y):
        try:
            logging.info("Split training and test input data")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

            # Best model and parameters are found using RandomizedSearchCV and Optuna
            # Check notebook for details.
            best_params = {'iterations': 987,
            'learning_rate': 0.00794812040578884,
            'depth': 10,
            'l2_leaf_reg': 1.8797385861143014e-08,
            'subsample': 0.7495072912757781,
            'colsample_bylevel': 0.8787194065417225,
            'min_data_in_leaf': 75,
            'od_type': 'IncToDec',
            'od_wait': 47}

            model = CatBoostClassifier(**best_params, silent=True)
            
            logging.info("Training the model with best parameters")
            model.fit(X_train, y_train)

            logging.info("Saving the trained model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Predicting on test data")
            predicted = model.predict(X_test)
            
            print(classification_report(y_test, predicted))
        except Exception as e:
            logging.error("An error occurred during model training", exc_info=True)
            raise CustomException(e, sys)