import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.utils import save_object, model_evaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data and target data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Support Vector Regressor": SVR()
            }

            model_report: dict= model_evaluation(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models) 

            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square            

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(e, sys)