import pickle

import pandas as pd


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
        cat_cols: list[str] | None
    ) -> dict:
        return {}


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
