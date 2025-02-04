import pickle 
import numpy as np
import pandas as pd
import os,sys
from src.Creditcardfaultdetection.exception import custom_exception
from src.Creditcardfaultdetection.logger import logging

from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info("error occured in save object")
        raise custom_exception(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        model.fit(X_train,y_train)
        logging.info('model training is done')
        
        y_pred= model.predict(X_test)
        logging.info('model prediction is done')
        
        accuracy=np.round(accuracy_score(y_test,y_pred),2)
        logging.info(f'the accuracy is {accuracy} for {model}')
        
        
        report=f'The accuracy for {model} is {accuracy}'
        
        return (
            report
        ) 
        
        
    except Exception as e:
        
        logging.info('error occured in the evaluate model')
        raise custom_exception(e,sys)
    
    
    
def load_obj(file_path):
    try :
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)   
        
    except Exception as e:
        logging.info('error occured in the load object ')
        raise custom_exception(e,sys)

