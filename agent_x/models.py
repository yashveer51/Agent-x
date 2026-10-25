import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class MajorityClassClassifier(BaseEstimator, ClassifierMixin):
    """Simple baseline model that always predicts the majority class from training data."""
    def fit(self, X, y):
        self.majority_class_ = pd.Series(y).mode()[0]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_)

    def predict_proba(self, X):
        probs = np.zeros((len(X), len(self.classes_)))
        idx = np.where(self.classes_ == self.majority_class_)[0][0]
        probs[:, idx] = 1.0
        return probs

class ModelFactory:
    """Factory class to instantiate baseline and machine learning classifiers."""

    @staticmethod
    def get_model(model_type: str = "random_forest", random_state: int = 42, **kwargs) -> Any:
        if model_type == "random_forest":
            params = {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10, "random_state": random_state}
            params.update(kwargs)
            return RandomForestClassifier(**params)
        elif model_type == "logistic_regression":
            params = {"random_state": random_state, "max_iter": 1000}
            params.update(kwargs)
            return LogisticRegression(**params)
        elif model_type == "majority_class":
            return MajorityClassClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
