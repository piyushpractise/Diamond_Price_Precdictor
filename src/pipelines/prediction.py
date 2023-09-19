import os
import sys
import pandas as pd
from src.log import logging
from src.exception import CustomException
from src.utils import save_obj, evaluate, load_object

class PredictingPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('storage','preprocessor.pkl')
            model_path = os.path.join('storage', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('Exception occured in Prediction Pipelines')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, carat:float, depth:float, table:float, x:float, y:float, z:float, cut:str,color:str,clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.clarity = clarity
        self.color = color

    def get_data(self):
        try:
            custom_data = {'x':[self.x],'y':[self.y],'z':[self.z],
            'carat':[self.carat], 'depth':[self.depth], 'carat':[self.table],
            'cut':[self.cut], 'color':[self.color], 'clarity':[self.clarity]}
            df = pd.DataFrame(Custom_data)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception occured while loading data from user')
            raise CustomException(e,sys)