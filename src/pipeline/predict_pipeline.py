import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'
        
    def predict(self, features):
        try:
            logging.info("Loading the preprocessor and model...")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            logging.info("Preprocessing the input features...")
            data_scaled = preprocessor.transform(features)
            logging.info(f"Transformed features: {data_scaled}")
            
            logging.info("Making predictions...")
            preds = model.predict(data_scaled)
            
            logging.info(f"Predictions completed: {preds}")
            return preds

        except Exception as e:
            raise CustomException(e, sys)
