import os
import sys
import pickle
import numpy as ny
import pandas as pd
from src.log import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_obj(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info('Error in saving pickle file')
        raise CustomException(e,sys)

def evaluate(x_train,y_train,x_test,y_test,models):
    try:
        result = {}
        for i in range(len(models)):
            m = list(models.values())[i]
            m.fit(x_train,y_train) # Training Model
            y_pred = m.predict(x_test) # Predicting Test Data
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result[list(models.keys())[i]] = mae, mse, r2
        return result

    except Exception as e:
        logging.info('Error in evaluatng model')
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Error occured while loading object from utils')
        raise CustomException(e,sys)