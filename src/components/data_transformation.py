import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['Tenure Months', 'Monthly Charges', 'Total Charges']
            categorical_columns = [
                'Gender',
                'Senior Citizen',
                'Partner',
                'Dependents',
                'Phone Service',
                'Multiple Lines',
                'Internet Service',
                'Online Security',
                'Online Backup',
                'Device Protection',
                'Tech Support',
                'Streaming TV',
                'Streaming Movies',
                'Contract',
                'Paperless Billing',
                'Payment Method',
            ]

            num_pipeline = Pipeline(
                steps= [("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps= [("onehotencoder", OneHotEncoder(drop="first"))]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_transformer_object()

            train_df = preprocessor_obj.fit_transform(train_df)
            test_df = preprocessor_obj.transform(test_df)

            logging.info("Preprocessing done.")

            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)
            X = pd.concat([train_df.iloc[:, :-1], test_df.iloc[:, :-1]])
            y = pd.concat([train_df.iloc[:, -1], test_df.iloc[:, -1]])
            
            logging.info("Applying SMOTE on training data.")
            smote = SMOTE(sampling_strategy='minority')
            X, y = smote.fit_resample(X, y)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                X,
                y,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)