import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception
from sklearn.impute import SimpleImputer
from src.Creditcardfaultdetection.utils.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.datatrasnconfig = DataTransformationconfig()
        
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            num_features = ["LIMIT_BAL"]
            cat_features = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

        # Correcting the categorical feature encoding
            pay_categories = [
                ["no consumption", "pay duly", "payment on time", "payment delay for one month",
                "payment delay for two months", "payment delay for three months",
                "payment delay for four months", "payment delay for five months",
                "payment delay for six months", "payment delay for seven months",
                "payment delay for eight months"]
                ] * len(cat_features)  # Repeat the same categories for all PAY_* columns

        # Define numerical pipeline
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # Fix order
                ("scaler", StandardScaler())
            ])
            logging.info("Numerical Pipeline created")

        # Define categorical pipeline
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Fix order
                ("encoder", OrdinalEncoder(categories=pay_categories))
            ])
            logging.info("Categorical pipeline created")

        # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])
            logging.info("Pipeline successfully created")
        
            return preprocessor

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise custom_exception(e, sys)

        
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("data read successfully")
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column = "default.payment.next.month"
            drop_columns = [target_column,"AGE"]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("data preprocessed successfully")            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("train and test data ready")
            
            save_object(
                file_path = self.datatrasnconfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("File object save successfully ")
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            logging.info("Error occured in data transformation")
            raise custom_exception(e,sys)