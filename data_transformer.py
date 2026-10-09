import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, MinMaxScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor

from custom_transformers import on_columns, GroupStatImputer, EstimatorImputer, MyTargetEncoder


class DataTransformer:
    def __init__(self, *, config, path: str, model_name: str, verbose: int = 0):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.path = path
        self.model_name = model_name
        self.verbose = verbose
        self.state_checkpoint = {}  
        self.pipeline = self._build_pipeline()


    def _build_pipeline(self):
        seed = self.config.general.seed

        if self.model_name in ('LogisticRegression', 'LinearSVC', 'DLM'):
            pipeline = Pipeline([
                ('embarked_impute', on_columns(SimpleImputer(strategy='most_frequent'), ['Embarked'])),
                ('fare_impute', on_columns(SimpleImputer(strategy='median'), ['Fare'])),
                ('new_on_ticket', on_columns(MyTargetEncoder(group_cols=['Ticket'], name='Ticket'), ['Ticket'])),
                ('new_on_surname', on_columns(MyTargetEncoder(group_cols=['Surname', 'Pclass'], name='Surname'), ['Surname', 'Pclass'])),
                ('ticket_surname_drop', on_columns('drop', ['Ticket', 'Surname'])),
                ('onehot', on_columns(
                    OneHotEncoder(drop='if_binary', min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    ['Sex', 'Embarked', 'Title', 'Deck', 'FamilySize']
                )),
                ('age_impute', EstimatorImputer(target_col='Age', estimator=RandomForestRegressor(n_estimators=20, max_depth=5, max_samples=0.7, max_features=0.7, min_samples_leaf=3, random_state=seed))),
                ('age_scale', on_columns(PowerTransformer(method='box-cox'), ['Age'])),
                ('fare_scale', on_columns(QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=seed), ['Fare'])),
                ('sibsp_parch_scale', on_columns(MinMaxScaler(), ['SibSp', 'Parch'])),
                ('survivalrate_scale', on_columns(StandardScaler(), ['SurvivalRateTicket', 'SurvivalRateSurname']))
            ])
        elif self.model_name in ('DecisionTreeClassifier', 'RandomForestClassifier', 'CatBoostClassifier'):
            pipeline = Pipeline([
                ('embarked_impute', on_columns(SimpleImputer(strategy='most_frequent'), ['Embarked'])),
                ('fare_impute', on_columns(SimpleImputer(strategy='median'), ['Fare'])),
                ('new_on_ticket', on_columns(MyTargetEncoder(group_cols=['Ticket'], name='Ticket'), ['Ticket'])),
                ('new_on_surname', on_columns(MyTargetEncoder(group_cols=['Surname', 'Pclass'], name='Surname'), ['Surname', 'Pclass'])),
                ('ticket_surname_drop', on_columns('drop', ['Ticket', 'Surname'])),
                ('onehot', on_columns(
                    OneHotEncoder(drop='if_binary', min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    ['Sex', 'Embarked', 'Title', 'Deck', 'FamilySize']
                )),
                ('age_impute', EstimatorImputer(target_col='Age', estimator=RandomForestRegressor(n_estimators=20, max_depth=5, max_samples=0.7, max_features=0.7, min_samples_leaf=3, random_state=seed))),
            ])
        else:
            raise ValueError(f'Pipeline for {self.model_name} is not defined')

        pipeline.set_output(transform='pandas')
        return pipeline


    def _validate_output(self, *, X: pd.DataFrame, X_transformed: pd.DataFrame, is_fit: bool) -> None:
        if len(X_transformed) != len(X) or not X_transformed.index.equals(X.index):
            raise ValueError('Row count or index changed during transform')

        non_numeric = [c for c in X_transformed.columns if not pd.api.types.is_numeric_dtype(X_transformed[c])]
        if non_numeric:
            raise ValueError(f'Non-numeric columns after transform: {non_numeric}')

        values = X_transformed.to_numpy(dtype=float)
        not_finite = X_transformed.columns[~np.isfinite(values).all(axis=0)].tolist()
        if not_finite:
            raise ValueError(f'NaN or inf after transform in: {not_finite}')

        if is_fit:
            self.state_checkpoint['feature_names'] = list(X_transformed.columns)
            self.state_checkpoint['train_mean'] = X_transformed.mean()
            self.state_checkpoint['train_std'] = X_transformed.std()
        elif list(X_transformed.columns) != self.state_checkpoint['feature_names']:
            raise ValueError('Output columns differ from those seen at fit')

        if not self.verbose:
            return

        self.logger.info(f'{"Fit" if is_fit else "Transform"} shape change: {X.shape} -> {X_transformed.shape}')

        missing_i = {col: int(n) for col, n in X.isna().sum().items() if n}
        missing_o = {col: int(n) for col, n in X_transformed.isna().sum().items() if n}
        if missing_i or missing_o:
            self.logger.info(f'Missing on input: {missing_i}, on output: {missing_o}')

        if is_fit:
            self.logger.info(f'Features: {list(X_transformed.columns)}')
        else:
            train_mean = self.state_checkpoint['train_mean']
            train_std = self.state_checkpoint['train_std'].replace(0, np.nan)
            shift = ((X_transformed.mean() - train_mean) / train_std).abs()
            drifted = shift[shift > 0.5].round(2)
            if not drifted.empty:
                self.logger.warning(f'Drift mean: {drifted.to_dict()}')


    def _as_frame(self, X_transformed) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            raise TypeError(f'Pipeline must return DataFrame, got {type(X_transformed).__name__}')
        return X_transformed


    def fit(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> 'DataTransformer':
        self.fit_transform(X=X, y=y) # Not pipeline.fit, for MyTargetEncoder correct work
        return self


    def transform(self, *, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self._as_frame(self.pipeline.transform(X))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=False)
        return np.asarray(X_transformed, dtype=float)


    def fit_transform(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        X_transformed = self._as_frame(self.pipeline.fit_transform(X, y))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=True)
        return np.asarray(X_transformed, dtype=float)


    def get_feature_names(self) -> np.ndarray:
        return self.pipeline.get_feature_names_out()


    def save_state(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'wb') as f:
            pickle.dump(self.state_checkpoint, f)

        with open(os.path.join(self.path, 'pipeline.pickle'), 'wb') as f:
            pickle.dump(self.pipeline, f)


    def load_state(self) -> None:
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'rb') as f:
            self.state_checkpoint = pickle.load(f)

        with open(os.path.join(self.path, 'pipeline.pickle'), 'rb') as f:
            self.pipeline = pickle.load(f)
