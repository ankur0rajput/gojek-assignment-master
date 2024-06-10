from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        prediction = self.clf.predict(df_test[self.features].values)
        metric_name = 'roc_auc_score'
        metric_score = roc_auc_score(df_test[self.target].values, prediction)
        return {metric_name: metric_score}
        '''
        raise NotImplementedError(
            f"You're almost there! Identify an appropriate evaluation metric for your model and implement it here. "
            f"The expected output is a dictionary of the follow
        '''    
    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]