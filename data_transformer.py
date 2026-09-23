import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformer:
    def __init__(self, *, config, path: str, model_name: str):
        self.config = config
        self.path = path
        self.model_name = model_name
        self.state_checkpoint = {}
        self.column_transformer = None


    def fit(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('age', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), ['Age']),
                ('num_standard', StandardScaler(), ['Fare']),
                ('embarked', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first'))
                ]), ['Embarked']),
                ('cat_onehot', OneHotEncoder(drop='first'), ['Sex'])
            ],
            remainder='passthrough'
        )
        self.column_transformer.fit(X)


    def transform(self, *, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self.column_transformer.transform(X)
        return X_transformed


    def fit_transform(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X=X, y=y)
        return self.transform(X=X)


    def save_state(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'wb') as f:
            pickle.dump(self.state_checkpoint, f)

        with open(os.path.join(self.path, 'column_transformer.pickle'), 'wb') as f:
            pickle.dump(self.column_transformer, f)


    def load_state(self) -> None:
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'rb') as f:
            self.state_checkpoint = pickle.load(f)

        with open(os.path.join(self.path, 'column_transformer.pickle'), 'rb') as f:
            self.column_transformer = pickle.load(f)
