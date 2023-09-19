import os
import sys
import pandas as pd
from src.log import logging
from dataclasses import dataclass
from src.utils import save_obj, evaluate
from src.exception import CustomException
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


@dataclass
class TrainerConfig:
    model_ob_filepath = os.path.join('storage', 'model.pkl')

class Trainer:
    def __init__(self):
        self.model_config = TrainerConfig()

    def start_trainer(self, Train, Test):
        try:
            logging.info('Splitting datasets into X and Y variables')
            x_train,y_train,x_test,y_test = (Train[:,:-1],Train[:,-1],Test[:,:-1],Test[:,-1])

            models = {'LR':LinearRegression(),
            'LR with lasso':Lasso(),
            'LR with ridge':Ridge(),
            'LR with EN':ElasticNet(),
            'knn':KNeighborsRegressor(n_neighbors=5),
            'dt':DecisionTreeRegressor(),
            'Random Forest':RandomForestRegressor(),
            'Gradient Boosting':GradientBoostingRegressor(),           
            'xgb':XGBRegressor(),
            'lgb':LGBMRegressor(),}

            model_result:dict=evaluate(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,models=models)
            print(model_result)
            print('\n','='*250)
            logging.info(f'Model Report: {model_result}')

            # Selecting Best Model Score
            best_model_score = min(sorted(model_result.values()))
            best_model_name = list(model_result.keys())[list(model_result.values()).index(best_model_score)]
            best_model = models[best_model_name]
            print(f'Best Model Selected, Model Name:{best_model_name}, mae score is {best_model_score[0]}, mse score is {best_model_score[1]}, r2 score is {best_model_score[2]}')
            print('\n','='*250)
            logging.info(f'Best Model Selected, Model Name:{best_model_name}, mae score is {best_model_score[0]}, mse score is {best_model_score[1]}, r2 score is {best_model_score[2]}')
            
            save_obj(filepath=self.model_config.model_ob_filepath, obj=best_model)

        except Exception as e:
            logging.info('Error while training model')
            raise CustomException(e,sys)