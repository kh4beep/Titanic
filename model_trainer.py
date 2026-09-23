import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, *, config, path: str, model_name: str):
        self.config = config
        self.path = path
        self.model_name = model_name
        self.model = None


    def fit(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_cols: list[str] | None = None
    ) -> dict:
        artifacts = {}
        if self.model_name == 'LogisticRegression':
            self.model = LogisticRegression(random_state=self.config.general.seed)
        if self.model_name == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(random_state=self.config.general.seed)
        if self.model_name == 'RandomForestClassifier':
            self.model = RandomForestClassifier(random_state=self.config.general.seed)
        self.model.fit(X_train, y_train)
        return artifacts


    def predict(
        self,
        *,
        X: pd.DataFrame,
    ) -> pd.Series:
        return self.model.predict(X)


    def save_state(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.model, f)


    def load_state(self):
        with open(self.path, 'rb') as f:
            self.model = pickle.load(f)
