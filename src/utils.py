import os 
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)
    
def model_evaluation(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            gs = GridSearchCV(estimator=model, param_grid=params[model_name], cv=3, n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report

    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)