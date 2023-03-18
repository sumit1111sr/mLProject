import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTraninerConfig:
    trained_model_file_path:str = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTraninerConfig()
         
    def initiate_model_training(self,test_array,train_array):
        try:
            logging.info("Splitting Training Input Data")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]    
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decison Tree":DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor,
                "Linear Regression": LinearRegression(),
                "K-Negihbours Regressor":KNeighborsRegressor(),
                "XGBRegressor": XGBRFRegressor(),
                "Adaboost Regressor":AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best mdodel found")
            logging.info("Best model found on both test and train dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)