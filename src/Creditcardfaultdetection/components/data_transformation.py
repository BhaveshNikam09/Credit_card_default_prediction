import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.Creditcardfaultdetection.logger import logging
from src.Creditcardfaultdetection.exception import custom_exception
from sklearn.impute import SimpleImputer
from src.Creditcardfaultdetection.utils.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.datatrasnconfig = DataTransformationconfig()
        # Ensure the pay_categories list is consistent
        self.pay_categories = [
            "no consumption", "pay duly", "payment on time", "payment delay for one month",
            "payment delay for two months", "payment delay for three months", "payment delay for four months",
            "payment delay for five months", "payment delay for six months", "payment delay for seven months",
            "payment delay for eight months"
        ]
        
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            num_features = ["LIMIT_BAL"]
            cat_features = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

            # Define numerical pipeline
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            logging.info("Numerical Pipeline created")

            # Define categorical pipeline with consistent pay_categories
            cat_pipeline = Pipeline([
                ("encoder", OrdinalEncoder(categories=[self.pay_categories] * len(cat_features), handle_unknown='use_encoded_value', unknown_value=-1))
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

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data read successfully")

            # Debugging: Log the unique values in the PAY_* columns for test data
            for col in ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
                logging.info(f"Unique values in {col} in test data: {test_df[col].unique()}")

            preprocessing_obj = self.get_data_transformation()

            # Check for unknown categories in the test data
            for col in ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
                unknown_values = set(test_df[col].unique()) - set(self.pay_categories)  # Check against predefined categories
                if unknown_values:
                    logging.warning(f"Found unknown categories in {col}: {unknown_values}")

            target_column = "default.payment.next.month"
            drop_columns = [target_column,
                "ID", "AGE", "SEX", "EDUCATION", "MARRIAGE",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
            ]
            
            # Drop columns and separate target feature
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Fit and transform training data, and transform test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Data preprocessed successfully")
            
            # Combine features with target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Train and test data ready")
            
            # Save the preprocessor object
            save_object(
                file_path=self.datatrasnconfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("File object saved successfully")
            return train_arr, test_arr

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise custom_exception(e, sys)
