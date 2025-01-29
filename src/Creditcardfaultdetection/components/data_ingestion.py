import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception

@dataclass
class DataIngestionconfig:
    raw_data_path: str=os.path.join("artifacts","raw.csv")
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    
class DataIngestion:
    def __init__(self):
        self.config=DataIngestionconfig()
        
    def read_data(self):
        try:
            logging.info("Reading raw data")
            
            file_path=Path('notebooks/data/Credit_Card.csv')
            data=pd.read_csv(file_path)
            logging.info("Raw data read successfully")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            data.to_csv(self.config.raw_data_path,index=False)

            logging.info(f"Raw data saved at {self.config.raw_data_path}")
            
            train_data,test_data=train_test_split(data,test_size=0.2,random_state=42) 
            
            train_data.to_csv(self.config.train_data_path,index=False)
            logging.info(f"Train data saved at {self.config.train_data_path}")
            
             
            test_data.to_csv(self.config.test_data_path,index=False)
            logging.info(f"Test data saved at {self.config.test_data_path}")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
            
                       
        except Exception as e:
            logging.error(f"Error in reading raw data: {str(e)}")
            raise custom_exception.DataIngestionException("Error in reading raw data")