import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate and log metrics: confusion matrix, precision, recall, f1 score.
        """
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Handling binary classification confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = None
        
        metrics = {
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
        
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"True Positives: {tp}")
        logging.info(f"True Negatives: {tn}")
        logging.info(f"False Positives: {fp}")
        logging.info(f"False Negatives: {fn}")
        
        return metrics

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

            # Stacking Classifier
            base_learners = [
                ('xgb', XGBClassifier()),
                ('rf', RandomForestClassifier()),
                ('lr', LogisticRegression(max_iter=1000))
            ]
            stacking_model = StackingClassifier(
                estimators=base_learners,
                final_estimator=LogisticRegression(max_iter=1000)
            )
            models["StackingClassifier"] = (stacking_model, {})

            best_model = None
            best_score = 0
            best_metrics = {}

            for model_name, (model, param_grid) in models.items():
                logging.info(f"Tuning {model_name}")
                tuned_model = self.tune_and_train_model(model, param_grid, X_train, y_train)

                # Evaluate the model using cross-validation
                cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=3, scoring='accuracy')
                mean_cv_score = cv_scores.mean()

                # Evaluate the model on the test set
                predicted = tuned_model.predict(X_test)
                test_score = accuracy_score(y_test, predicted)

                # Calculate and log additional metrics
                metrics = self.calculate_metrics(y_test, predicted)

                logging.info(f"{model_name} cross-validated accuracy: {mean_cv_score}")
                logging.info(f"{model_name} test accuracy: {test_score}")

                if test_score > best_score:
                    best_score = test_score
                    best_model = tuned_model
                    best_metrics = metrics

            if best_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model.__class__.__name__} with score {best_score}")
            logging.info(f"Best model metrics: {best_metrics}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_score, best_metrics

        except Exception as e:
            raise CustomException(e, sys)
