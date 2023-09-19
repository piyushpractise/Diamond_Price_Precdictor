import os
import sys
import pandas as pd
from src.log import logging
from src.exception import CustomException
from src.components.ingection import Ingestion
from src.components.transform import Transformation
from src.components.trainer import Trainer

if __name__=='__main__':
    # Data Ingestion
    di = Ingestion()
    train_path, test_path = di.start_ingestion()
    print(train_path,test_path)

    # Data Transformation
    dt = Transformation()
    train, test, _ = dt.start_transformation(train_path, test_path)

    # Model Selection and Training
    model_train = Trainer()
    model_train.start_trainer(train, test)