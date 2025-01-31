import pandas as pd
from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception
from src.Creditcardfaultdetection.components.data_ingestion import DataIngestion
from src.Creditcardfaultdetection.components.data_transformation import DataTransformation
from src.Creditcardfaultdetection.components.model_trainer import model_trainer

obj = DataIngestion()
train_data_path,test_data_path=obj.read_data()

obj1 = DataTransformation()
train_arr,test_arr = obj1.initialize_data_transformation(train_data_path,test_data_path)

model=model_trainer()
report=model.initiate_model_training(train_arr,test_arr)

print(report)