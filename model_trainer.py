import pickle

import numpy as np
import pandas as pd

from typing import Any, cast
from omegaconf import OmegaConf
from scipy.stats import norm, loguniform, uniform, randint
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from catboost import CatBoostClassifier, Pool

from dl_model import DLM
from utils import LogUniformInt


class ModelTrainer:
    def __init__(self, *, config, path: str, model_name: str, best_params: dict | None = None):
        self.config = config
        self.params = config.models.params[model_name]
        self.path = path
        self.model_name = model_name
        self.best_params = best_params or {}
        self.model: Any = self._build_model()


    def _build_model(self) -> Any:
        kwargs = cast(dict[str, Any], OmegaConf.to_container(self.params.kwargs, resolve=True))
        kwargs.update(self.best_params)

        model: Any
        if self.model_name == 'DLM':
            model = DLM(**kwargs)
        elif self.model_name == 'LinearSVC':
            model = LinearSVC(**kwargs)
        elif self.model_name == 'SVC':
            model = SVC(**kwargs)
        elif self.model_name == 'LogisticRegression':
            model = LogisticRegression(**kwargs)
        elif self.model_name == 'KNeighborsClassifier':
            model = KNeighborsClassifier(**kwargs)
        elif self.model_name == 'GaussianNB':
            model = GaussianNB(**kwargs)
        elif self.model_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**kwargs)
        elif self.model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**kwargs)
        elif self.model_name == 'CatBoostClassifier':
            model = CatBoostClassifier(**kwargs)
        else:
            raise ValueError(f'Wrong model_name: {self.model_name}')

        self.base_model = model

        if self.params.get('is_calibration', False):
            model = CalibratedClassifierCV(estimator=model, **self.params.calibration_kwargs)

        if not hasattr(model, 'predict_proba'):
            raise ValueError(f'{self.model_name} has no predict_proba')
        return model


    def _make_distributions(self, spec) -> dict:
        result = {}
        for name, d in spec.items():
            if d.type == 'choice':
                result[name] = list(d['values'])
            elif d.type == 'loguniform':
                result[name] = loguniform(d.a, d.b)
            elif d.type == 'uniform':
                result[name] = uniform(loc=d.a, scale=d.b - d.a)
            elif d.type == 'int':
                result[name] = randint(d.a, d.b + 1)
            elif d.type == 'norm':
                result[name] = norm(loc=d.a, scale=d.b)
            elif d.type == 'logint':
                result[name] = LogUniformInt(d.a, d.b)
            else:
                raise ValueError(f'Unknown distribution type: {d.type}')
        return result


    def _distance(self, params: dict, prior: dict, spec) -> float:
        d = 0.0
        for name, value in params.items():
            p, s = prior[name], spec[name]
            if s.type == 'choice':
                step = float(value != p)
            elif s.type in ('loguniform', 'logint'):
                step = abs(np.log(value / p)) / np.log(s.b / s.a)
            elif s.type in ('uniform', 'int'):
                step = abs(value - p) / (s.b - s.a)
            elif s.type == 'norm':
                step = abs(value - p) / s.b
            else:
                raise ValueError(f'Unknown distribution type: {s.type}')
            d += min(step, 1.0)
        return d


    def _closest_to_prior(self, cv_results: dict) -> int:
        mean = np.asarray(cv_results['mean_test_score'])
        std = np.asarray(cv_results['std_test_score'])
        n_splits = self.params.hyperparameter_search_kwargs.cv
        best = int(np.nanargmax(mean))
        threshold = mean[best] - std[best] / np.sqrt(n_splits)
        candidates = np.flatnonzero(mean >= threshold)

        spec = self.params.hyperparameter_search_kwargs.param_distributions
        prior = self.base_model.get_params()
        return int(min(candidates, key=lambda i: self._distance(cv_results['params'][i], prior, spec)))


    def get_best_params(self, *, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> dict:
        fit_kwargs: dict[str, Any] = self.params.get('fit_kwargs', {})
        if fit_kwargs:
            fit_kwargs = OmegaConf.to_container(fit_kwargs, resolve=True) # type: ignore[override]
        s = self.params.hyperparameter_search_kwargs
        search = RandomizedSearchCV(
            self.base_model,
            param_distributions=self._make_distributions(s.param_distributions),
            n_iter=s.n_iter,
            scoring=s.scoring,
            cv=StratifiedKFold(n_splits=s.cv, shuffle=True, random_state=self.config.general.seed),
            refit=lambda cv_results: self._closest_to_prior(cv_results),
            random_state=self.config.general.seed,
        )
        search.fit(X, y, **fit_kwargs)
        return search.best_params_


    def fit(self, *, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> dict:
        fit_kwargs: dict[str, Any] = self.params.get('fit_kwargs', {})
        if fit_kwargs:
            fit_kwargs = OmegaConf.to_container(fit_kwargs, resolve=True) # type: ignore[override]
        if self.model_name == 'CatBoostClassifier':
            if self.params.get('val_size', False):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.params.val_size, stratify=y, random_state=self.config.general.seed
                )
                self.model.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_kwargs)
            else:
                self.model.fit(X, y, **fit_kwargs)
        else:
            self.model.fit(X, y, **fit_kwargs)
        if self.model_name == 'DLM':
            return self.model.get_training_artifacts()
        return {}


    def predict_proba(self, *, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


    def get_feature_importance(self) -> dict[str, np.ndarray]:
        inner = (
            [cc.estimator for cc in self.model.calibrated_classifiers_]
            if isinstance(self.model, CalibratedClassifierCV) else [self.model]
        )
        if self.model_name in ('LogisticRegression', 'LinearSVC'):
            return {'coef': np.mean([m.coef_[0] for m in inner], axis=0)}
        if self.model_name in ('DecisionTreeClassifier', 'RandomForestClassifier'):
            return {'impurity': np.mean([m.feature_importances_ for m in inner], axis=0)}
        if self.model_name == 'CatBoostClassifier':
            return {'prediction_values_change': np.mean([m.get_feature_importance() for m in inner], axis=0)}
        return {}


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
