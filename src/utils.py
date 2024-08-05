import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

from sklearn.metrics import r2_score

#f#rom sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_report[model_name] = score
        return model_report
    except Exception as e:
        raise CustomException(e, sys)


   
