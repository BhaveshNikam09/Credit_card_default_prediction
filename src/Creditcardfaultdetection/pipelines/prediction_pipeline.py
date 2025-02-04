import os
import sys
import pandas as pd
from src.Creditcardfaultdetection.exception import custom_exception
from src.Creditcardfaultdetection.utils.utils import load_obj
from src.Creditcardfaultdetection.logger import logging

class predictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
        
        # Load preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor_obj = load_obj(preprocessor_path)
            model = load_obj(model_path)
        # Apply the stored preprocessor (includes ordinal encoding)
            transformed_features = preprocessor_obj.transform(features)

        # Predict using the trained model
            prediction = model.predict(transformed_features)

            if prediction == 0:
                 return "No payment default detected for this user."
            else:
                return "Alert: Payment default detected for this user."

                    
        except Exception as e:
            raise custom_exception(e, sys)


class custom_data:
    def __init__(self,
                LIMIT_BAL: int,
                PAY_1: str,
                PAY_2: str,
                PAY_3: str,
                PAY_4: str,
                PAY_5: str,
                PAY_6: str,
    ):
        self.LIMIT_BAL = LIMIT_BAL
        self.PAY_1 = PAY_1
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
        
    def get_data_as_df(self):
        try:
            custom_data_input = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "PAY_1": [self.PAY_1],
                "PAY_2": [self.PAY_2],
                "PAY_3": [self.PAY_3],
                "PAY_4": [self.PAY_4],
                "PAY_5": [self.PAY_5],
                "PAY_6": [self.PAY_6],
            }

            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise custom_exception(e, sys)
        
        

