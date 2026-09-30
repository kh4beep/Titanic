import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool

from dl_model import DLM


class ModelTrainer:
    def __init__(self, *, config, path: str, model_name: str):
        self.config = config
        self.path = path
        self.model_name = model_name
        self.model = None


    def fit(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cat_cols: list[str] | None = None
    ) -> dict:
        artifacts = {}
        if self.model_name == 'DLM':
            self.model = DLM()
            self.model.fit(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
        else:
            if self.model_name == 'LogisticRegression':
                self.model = LogisticRegression(random_state=self.config.general.seed)
            elif self.model_name == 'DecisionTreeClassifier':
                self.model = DecisionTreeClassifier(random_state=self.config.general.seed)
            elif self.model_name == 'RandomForestClassifier':
                self.model = RandomForestClassifier(random_state=self.config.general.seed)
            elif self.model_name == 'CatBoostClassifier':
                self.model = CatBoostClassifier(random_state=self.config.general.seed)
            else:
                raise Exception('Wrong model_name:', self.model_name)
            
            self.model.fit(X_train, y_train)
        return artifacts


    def predict(
        self,
        *,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise Exception('Model is not defined')
        return self.model.predict(X)


    def save_state(self):
        if type(self.model) == DLM:
            state = self.model.get_state()
            with open(self.path, "wb") as f:
                pickle.dump(state, f)
        else:
            with open(self.path, "wb") as f:
                pickle.dump(self.model, f)


    def load_state(self):
        if self.model_name == 'DLM':
            with open(self.path, 'rb') as f:
                state = pickle.load(f)
            self.model = DLM()
            self.model.put_state(state=state)
        else:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
