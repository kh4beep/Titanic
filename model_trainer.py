import pickle

import numpy as np
import pandas as pd

from typing import Any, cast
from omegaconf import OmegaConf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
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
        self.model: Any = None


    def _build_model(self) -> Any:
        p = self.config.models.params[self.model_name]
        kwargs = cast(dict[str, Any], OmegaConf.to_container(p.kwargs, resolve=True))

        model: Any
        if self.model_name == 'DLM':
            model = DLM(**kwargs)
        elif self.model_name == 'LinearSVC':
            model = LinearSVC(**kwargs)
        elif self.model_name == 'LogisticRegression':
            model = LogisticRegression(**kwargs)
        elif self.model_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**kwargs)
        elif self.model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**kwargs)
        elif self.model_name == 'CatBoostClassifier':
            model = CatBoostClassifier(**kwargs)
        else:
            raise ValueError(f'Wrong model_name: {self.model_name}')

        if p.get('is_calibration', False):
            model = CalibratedClassifierCV(estimator=model, **p.calibration_kwargs)

        if not hasattr(model, 'predict_proba'):
            raise ValueError(f'{self.model_name} has no predict_proba')
        return model


    def fit(self, *, X_train, y_train, X_val, y_val) -> dict:
        if self.model is None:
            self.model = self._build_model()
        if self.model_name == 'DLM':
            self.model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        else:
            self.model.fit(X_train, y_train)
        return {}


    def predict_proba(self, *, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


    def save_state(self) -> None:
        if self.model_name == 'DLM':
            self.model.save(path=self.path)
        else:
            with open(self.path, 'wb') as f:
                pickle.dump(self.model, f)


    def load_state(self) -> None:
        if self.model_name == 'DLM':
            self.model = DLM.load(path=self.path)
        else:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
