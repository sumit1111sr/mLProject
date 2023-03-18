import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransormation,DataTransformationConfig
from src.components.model_trainer import ModelTraninerConfig,ModelTrainer
@dataclass
class DataIngestionConfig:
    data_train_path:str =os.path.join('artifact','train.csv')
    data_test_path:str =os.path.join('artifact','test.csv')
    data_raw_path:str =os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the data as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.data_train_path),exist_ok=True)

            df.to_csv(self.ingestion_config.data_raw_path,index=False)
            logging.info("Train test initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.data_train_path,index=False)
            test_set.to_csv(self.ingestion_config.data_test_path,index=False)
            logging.info("Data Ingestion completed")
            return(
                self.ingestion_config.data_test_path,
                self.ingestion_config.data_train_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_trasformation = DataTransormation()
    train_array, test_array,_  = data_trasformation.initiate_data_transformation(train_data,test_data)
    model_trainer = ModelTrainer()
    
    print(model_trainer.initiate_model_training(train_array, test_array))




# src/components/data_ingestion.py