import os
import sys
import numpy as ny
import pandas as pd
from src.log import logging
from src.utils import save_obj
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

@dataclass
class TransformConfig:
    preprocessor_ob_filepath = os.path.join('storage', 'preprocessor.pkl')

class Transformation:
    def __init__(self):
        self.transformConfig = TransformConfig()
    def get_transformation_obj(self):
        try:
            logging.info('Data Transformation Initiated')
            # Separating Categorical & Numerical columns
            cat_col = ['cut', 'color', 'clarity']
            num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
            # Defining rankings for Ordinal Variable
            Rcut=["Fair","Good","Very Good","Premium","Ideal"]
            Rclarity = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            Rcolor = ["D", "E", "F", "G", "H", "I", "J"]

            logging.info('Pipeline Initiated')
            # Numerical Pipeline
            Pnum = Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),('Scaler',StandardScaler())])
            # Categorical Pipeline
            Pcat = Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),
                       ('Ordinal Encoder',OrdinalEncoder(categories=[Rcut,Rcolor,Rclarity])),
                       ('Scaler',StandardScaler())])
            com = ColumnTransformer(transformers=[('Num Pipeline',Pnum,num_col), ('Cat Pipeline',Pcat,cat_col)])

            return com

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)

    def start_transformation(self, train_path, test_path):
        try:
            # Reading data
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            logging.info('Reading of Train and Test data completed')
            logging.info(f'Train Dataframe:\n{train.head().to_string()}')
            logging.info(f'Test Dataframe:\n{test.head().to_string()}')

            logging.info('Obtaining Preprocessing Object')
            com_obj = self.get_transformation_obj()

            # Omitting unnesscary columns
            target = 'price'
            drop = [target,'id']
            x_train = train.drop(columns=drop, axis=1)
            y_train = train[target]
            x_test = test.drop(columns=drop, axis=1)
            y_test = test[target]

            # Applying transformation
            X_train = com_obj.fit_transform(x_train)
            X_test = com_obj.transform(x_test)
            logging.info("Applying preprocessing object on datasets.")
            Train = ny.c_[X_train, ny.array(y_train)]
            Test = ny.c_[X_test, ny.array(y_test)]

            save_obj(filepath=self.transformConfig.preprocessor_ob_filepath, obj=com_obj)
            logging.info('Processor pickle in created and saved')
            return (Train, Test, self.transformConfig.preprocessor_ob_filepath)

        except Exception as e:
            logging.info('Error while pickling processor')
            raise CustomException(e,sys)