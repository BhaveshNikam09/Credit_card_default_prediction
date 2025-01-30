import pickle 
import numpy as np
import pandas as pd
import os,sys
from src.Creditcardfaultdetection.exception import custom_exception
from src.Creditcardfaultdetection.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info("error occured in save object")
        raise custom_exception(e,sys)
