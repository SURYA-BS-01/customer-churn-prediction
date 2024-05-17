import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                gender: str,
                tenure_months: int,
                monthly_charges: float,
                total_charges: float,
                senior_citizen: str,
                partner: str,
                dependents: str,
                phone_service: str,
                multiple_lines: str,
                internet_service: str,
                online_security: str,
                online_backup: str,
                device_protection: str,
                tech_support: str,
                streaming_tv: str,
                streaming_movies: str,
                contract: str,
                paperless_billing: str,
                payment_method: str):
        self.gender = gender
        self.tenure_months = tenure_months
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges
        self.senior_citizen = senior_citizen
        self.partner = partner
        self.dependents = dependents
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Tenure Months": [self.tenure_months],
                "Monthly Charges": [self.monthly_charges],
                "Total Charges": [self.total_charges],
                "Senior Citizen": [self.senior_citizen],
                "Partner": [self.partner],
                "Dependents": [self.dependents],
                "Phone Service": [self.phone_service],
                "Multiple Lines": [self.multiple_lines],
                "Internet Service": [self.internet_service],
                "Online Security": [self.online_security],
                "Online Backup": [self.online_backup],
                "Device Protection": [self.device_protection],
                "Tech Support": [self.tech_support],
                "Streaming TV": [self.streaming_tv],
                "Streaming Movies": [self.streaming_movies],
                "Contract": [self.contract],
                "Paperless Billing": [self.paperless_billing],
                "Payment Method": [self.payment_method]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
