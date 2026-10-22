import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from custom_transformers import on_columns, GroupStatImputer, EstimatorImputer, MyTargetEncoder


class DataTransformer:
    def __init__(self, *, config, path: str, model_name: str, verbose: int = 0):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.path = path
        self.model_name = model_name
        self.verbose = verbose
        self.is_as_frame = model_name == 'CatBoostClassifier'
        self.state_checkpoint = {}  
        self.pipeline = self._build_pipeline()


    def _build_pipeline(self):
        seed = self.config.general.seed
        age_model = make_pipeline(
            ColumnTransformer(
                [
                    ('cat', OneHotEncoder(drop='if_binary', min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False), ['Title', 'Embarked', 'Deck', 'Sex']),
                    ('drop', 'drop', ['Ticket', 'Surname']),
                ],
                remainder='passthrough',
            ),
            RandomForestRegressor(n_estimators=20, max_depth=5, max_samples=0.7, max_features=0.7, min_samples_leaf=3, random_state=seed),
        )

        if self.model_name in ('LinearSVC', 'SVC', 'LogisticRegression', 'KNeighborsClassifier', 'GaussianNB', 'DLM'):
            pipeline = Pipeline([
                ('embarked_impute', on_columns(SimpleImputer(strategy='most_frequent'), ['Embarked'])),
                ('fare_impute', on_columns(SimpleImputer(strategy='median'), ['Fare'])),
                ('age_impute', EstimatorImputer(target_col='Age', estimator=age_model)),
                ('new_on_ticket', on_columns(MyTargetEncoder(group_cols=['Ticket'], name='Ticket'), ['Ticket'])),
                ('new_on_surname', on_columns(MyTargetEncoder(group_cols=['Surname', 'Pclass'], name='Surname'), ['Surname', 'Pclass'])),
                ('onehot_freq', on_columns(
                    OneHotEncoder(drop='first', sparse_output=False),
                    ['Sex', 'Embarked', 'Title']
                )),
                ('onehot_rary', on_columns(
                    OneHotEncoder(min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    ['Deck']
                )),
                ('age_transform', on_columns(PowerTransformer(method='box-cox'), ['Age'])),
                ('fare_transform', on_columns(QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=seed), ['Fare'])),
                ('sibsp_parch_transform', on_columns(FunctionTransformer(np.log1p), ['SibSp', 'Parch'])),
                ('scale', on_columns(StandardScaler(), ['SurvivalRateTicket', 'SurvivalRateSurname', 'Fare', 'SibSp', 'Parch'])),
                ('drop', on_columns('drop', ['Ticket', 'Surname'])),
            ])
        elif self.model_name in ('DecisionTreeClassifier', 'RandomForestClassifier'):
            pipeline = Pipeline([
                ('embarked_impute', on_columns(SimpleImputer(strategy='most_frequent'), ['Embarked'])),
                ('fare_impute', on_columns(SimpleImputer(strategy='median'), ['Fare'])),
                ('age_impute', EstimatorImputer(target_col='Age', estimator=age_model)),
                ('new_on_ticket', on_columns(MyTargetEncoder(group_cols=['Ticket'], name='Ticket'), ['Ticket'])),
                ('new_on_surname', on_columns(MyTargetEncoder(group_cols=['Surname', 'Pclass'], name='Surname'), ['Surname', 'Pclass'])),
                ('onehot_freq', on_columns(
                    OneHotEncoder(drop='first', sparse_output=False),
                    ['Sex', 'Embarked', 'Title']
                )),
                ('onehot_rary', on_columns(
                    OneHotEncoder(min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    ['Deck']
                )),
                ('drop', on_columns('drop', ['Ticket', 'Surname'])),
            ])
        elif self.model_name in ('CatBoostClassifier',):
            pipeline = Pipeline([
                ('embarked_impute', on_columns(SimpleImputer(strategy='most_frequent'), ['Embarked'])),
                ('fare_impute', on_columns(SimpleImputer(strategy='median'), ['Fare'])),
                ('age_impute', EstimatorImputer(target_col='Age', estimator=age_model)),
            ])
        else:
            raise ValueError(f'Pipeline for {self.model_name} is not defined')

        pipeline.set_output(transform='pandas')
        return pipeline


    def _validate_output(self, *, X: pd.DataFrame, X_transformed: pd.DataFrame, is_fit: bool) -> None:
        if len(X_transformed) != len(X) or not X_transformed.index.equals(X.index):
            raise ValueError('Row count or index changed during transform')

        numeric_cols = [c for c in X_transformed.columns if pd.api.types.is_numeric_dtype(X_transformed[c])]
        categorical_cols = [c for c in X_transformed.columns if c not in numeric_cols]

        if categorical_cols and not self.is_as_frame:
            raise ValueError(f'Non-numeric columns after transform: {categorical_cols}')

        numeric = X_transformed[numeric_cols].to_numpy(dtype=float)
        has_inf = np.isinf(numeric).any(axis=0)
        has_nan = np.isnan(numeric).any(axis=0)
        bad_numeric = [c for c, inf, nan in zip(numeric_cols, has_inf, has_nan) if inf or (nan and not self.is_as_frame)]
        if bad_numeric:
            raise ValueError(f'NaN or inf after transform in: {bad_numeric}')

        cat_with_nan = [c for c in categorical_cols if X_transformed[c].isna().any()]
        if cat_with_nan:
            raise ValueError(f'NaN in categorical columns after transform: {cat_with_nan}')
        cat_not_str = [c for c in categorical_cols if not X_transformed[c].map(lambda v: isinstance(v, str)).all()]
        if cat_not_str:
            raise ValueError(f'Categorical columns must contain only strings: {cat_not_str}')

        if is_fit:
            self.state_checkpoint['feature_names'] = list(X_transformed.columns)
            self.state_checkpoint['categorical_cols'] = categorical_cols
            self.state_checkpoint['train_categories'] = {c: set(X_transformed[c].unique()) for c in categorical_cols}
            self.state_checkpoint['train_mean'] = X_transformed[numeric_cols].mean()
            self.state_checkpoint['train_std'] = X_transformed[numeric_cols].std()
        else:
            if list(X_transformed.columns) != self.state_checkpoint['feature_names']:
                raise ValueError('Output columns differ from those seen at fit')
            if categorical_cols != self.state_checkpoint['categorical_cols']:
                raise ValueError(
                    f'Categorical columns differ from fit: {categorical_cols} vs {self.state_checkpoint["categorical_cols"]}'
                )

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
            shift = ((X_transformed[numeric_cols].mean() - train_mean) / train_std).abs()
            drifted = shift[shift > 0.5].round(2)
            if not drifted.empty:
                self.logger.warning(f'Drift mean: {drifted.to_dict()}')

            unseen = {}
            for c, known in self.state_checkpoint['train_categories'].items():
                share = float((~X_transformed[c].isin(known)).mean())
                if share > 0:
                    unseen[c] = round(share, 3)
            if unseen:
                self.logger.warning(f'Share of categories unseen at fit: {unseen}')


    def _as_frame(self, X_transformed) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            raise TypeError(f'Pipeline must return DataFrame, got {type(X_transformed).__name__}')
        return X_transformed


    def fit(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> 'DataTransformer':
        self.fit_transform(X=X, y=y) # Not pipeline.fit, for MyTargetEncoder correct work
        return self


    def transform(self, *, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        X_transformed = self._as_frame(self.pipeline.transform(X))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=False)
        return X_transformed if self.is_as_frame else np.asarray(X_transformed, dtype=float)


    def fit_transform(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame | np.ndarray:
        X_transformed = self._as_frame(self.pipeline.fit_transform(X, y))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=True)
        return X_transformed if self.is_as_frame else np.asarray(X_transformed, dtype=float)


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
