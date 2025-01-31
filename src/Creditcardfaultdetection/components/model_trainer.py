import os
import sys 
import pandas as pd
import numpy as np

from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception
from dataclasses import dataclass
from src.Creditcardfaultdetection.utils.utils import save_object,evaluate_model

from sklearn.ensemble import RandomForestClassifier

@dataclass
class CreditCardFaultDetectionConfig:
    model_path:str=os.path.join('artifacts','model.pkl')
    
class model_trainer:
    def __init__(self):
        self.model_train=CreditCardFaultDetectionConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('model training started')
            
            logging.info('spliting the data into the training and testing with target and independent features')
            
            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            
            logging.info('the hyperparameter tunning is already perform in the model_training.ipynb')
            params={'n_estimators': 300,
                    'max_depth': 7,
                    'min_samples_split': 8,
                    'min_samples_leaf': 9,
                    'max_features': 'sqrt'
                }
            RFC=RandomForestClassifier(**params,random_state=2)
            logging.info('the model is ready for the training')
            
            report=evaluate_model(X_train,y_train,X_test,y_test,RFC)
            logging.info(f'model trainig is done so the report is {report}')
            
            
            save_object(
                file_path=self.model_train.model_path,
                obj=RFC
            )
            
            return (
                report
            )
            
        except Exception as e :
            logging.info('error occured in the model trainig ')
            raise custom_exception(e,sys)