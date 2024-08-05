import os
import sys
from dataclasses import dataclass  # Add this import
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_and_train_model(self, model, param_grid, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
        return best_model

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "XGBClassifier": (XGBClassifier(), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7]
                }),
                "DecisionTreeClassifier": (DecisionTreeClassifier(), {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                "RandomForestClassifier": (RandomForestClassifier(), {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                }),
                "LogisticRegression": (LogisticRegression(max_iter=1000), {
                    'C': [0.01, 0.1, 1, 10]
                }),
                "GradientBoostingClassifier": (GradientBoostingClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                }),
                "AdaBoostClassifier": (AdaBoostClassifier(), {
                    'n_estimators': [50, 100, 200]
                }),
                "KNeighborsClassifier": (KNeighborsClassifier(), {
                    'n_neighbors': [3, 5, 7]
                }),
                "CatBoostClassifier": (CatBoostClassifier(verbose=False), {
                    'iterations': [100, 200],
                    'depth': [6, 10]
                }),
                "SVC": (SVC(), {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                })
            }

            best_model = None
            best_score = 0

            for model_name, (model, param_grid) in models.items():
                logging.info(f"Tuning {model_name}")
                tuned_model = self.tune_and_train_model(model, param_grid, X_train, y_train)

                # Evaluate the model using cross-validation
                cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=3, scoring='accuracy')
                mean_cv_score = cv_scores.mean()

                # Evaluate the model on the test set
                predicted = tuned_model.predict(X_test)
                test_score = accuracy_score(y_test, predicted)

                logging.info(f"{model_name} cross-validated accuracy: {mean_cv_score}")
                logging.info(f"{model_name} test accuracy: {test_score}")

                if test_score > best_score:
                    best_score = test_score
                    best_model = tuned_model

            if best_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model.__class__.__name__} with score {best_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
