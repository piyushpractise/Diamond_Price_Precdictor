import os
import sys
import pandas as pd
from src.log import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split

# Initialization of Data Ingection config
@dataclass
class IngectionConfig:
    train_path = os.path.join('storage','train.csv')
    test_path = os.path.join('storage','test.csv')
    raw_path = os.path.join('storage','raw.csv')

class Ingestion:
    def __init__(self):
        self.ingestion_config = IngectionConfig()
    
    def start_ingestion(self):
        logging.info('Data Ingestion Begins')
        try:
            df = pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            logging.info('Dataset has been read as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_path, index=False)

            logging.info("Training Test Split")
            train_set, test_set = train_test_split(df, test_size=0.35, random_state=47)
            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info("Ingestion of data is Completed")

            return(self.ingestion_config.train_path,self.ingestion_config.test_path)
        
        except Exception as e:
            logging.info('Error occured while Data Ingestion')