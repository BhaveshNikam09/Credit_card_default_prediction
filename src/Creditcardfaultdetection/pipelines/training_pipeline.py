import pandas as pd
from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception
from src.Creditcardfaultdetection.components.data_ingestion import DataIngestion

obj = DataIngestion()
train_data_path,test_data_path=obj.read_data()
