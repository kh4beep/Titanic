import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer


def on_columns(transformer, columns: list[str]) -> ColumnTransformer:
    """
    Apply a transformer to the given columns and pass the rest through unchanged.
    Note: ColumnTransformer puts the transformed columns first, so column order changes.
    Output names stay unprefixed (verbose_feature_names_out=False), so they must be unique.
    """
    return ColumnTransformer(
        [('t', transformer, columns)],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )



class GroupStatImputer(BaseEstimator, TransformerMixin):
    """
    Fill NaN in target_col with a per-group statistic learned at fit.
    Groups with fewer than min_count_group rows fall back to the global statistic.
    """
    def __init__(self, *, target_col: str, group_cols: list[str], method: str, min_count_group: int = 10) -> None:
        self.target_col = target_col
        self.group_cols = group_cols
        self.method = method
        self.min_count_group = min_count_group


    def fit(self, X: pd.DataFrame, y=None) -> 'GroupStatImputer':
        group_cols = list(self.group_cols)
        stats = X.groupby(group_cols)[self.target_col].agg([self.method, 'count'])
        self.group_stat_ = stats.loc[stats['count'] >= self.min_count_group, self.method]
        self.global_stat_ = X[self.target_col].agg(self.method)
        self.feature_names_in_ = np.asarray(X.columns)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'group_stat_')
        X = X.copy()
        group_cols = list(self.group_cols)
        fill = X[group_cols].merge(
            self.group_stat_, left_on=group_cols, right_index=True, how='left'
        )[self.method]
        X[self.target_col] = X[self.target_col].fillna(fill).fillna(self.global_stat_)
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_



class EstimatorImputer(BaseEstimator, TransformerMixin):
    """
    Fill NaN in target_col with predictions of an estimator trained on rows where it is known.
    The estimator sees every other column that reaches this step, so its feature set is
    defined by the pipeline position; wrap it in a Pipeline to encode or select inputs.
    """
    def __init__(self, *, target_col: str, estimator: BaseEstimator) -> None:
        self.target_col = target_col
        self.estimator = estimator


    def fit(self, X: pd.DataFrame, y=None) -> 'EstimatorImputer':
        X = X[X[self.target_col].notna()]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X.drop(columns=[self.target_col]), X[self.target_col])
        self.feature_names_in_ = np.asarray(X.columns)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'estimator_')
        X = X.copy()
        missing = X[self.target_col].isna()
        if missing.any():
            fill = self.estimator_.predict(X[missing].drop(columns=[self.target_col]))
            X.loc[missing, self.target_col] = fill
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_



class MyTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target rate per group, plus a flag whether the group was seen.
    rate = (sum + smoothing * prior) / (count + smoothing), so small groups shrink to the
    global rate. fit_transform cross-fits: each train row is encoded with stats from the
    other folds only, otherwise its own target would leak into its feature. transform uses
    stats from all fitted rows and is meant for unseen data only.
    """
    def __init__(self, *, group_cols: list[str], name: str, smoothing: float = 1.0) -> None:
        self.group_cols = group_cols
        self.name = name
        self.smoothing = smoothing


    @staticmethod
    def _stats(X: pd.DataFrame, y: np.ndarray, group_cols: list[str]) -> pd.DataFrame:
        return X[group_cols].assign(__y__=y).groupby(group_cols)['__y__'].agg(['count', 'sum'])


    def _encode(self, X: pd.DataFrame, stats: pd.DataFrame, prior: float) -> tuple[np.ndarray, np.ndarray]:
        joined = X[list(self.group_cols)].join(stats, on=list(self.group_cols))
        n = joined['count'].fillna(0).to_numpy()
        s = joined['sum'].fillna(0).to_numpy()
        rate = (s + self.smoothing * prior) / (n + self.smoothing)
        return (n > 0).astype(int), rate


    def fit(self, X: pd.DataFrame, y=None) -> 'MyTargetEncoder':
        if y is None:
            raise ValueError('MyTargetEncoder requires y')
        y = np.asarray(y, dtype=float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.stats_ = self._stats(X, y, list(self.group_cols))
        self.prior_ = float(y.mean())
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'stats_')
        known, rate = self._encode(X, self.stats_, self.prior_)
        return X.assign(**{f'IsGroup{self.name}': known, f'SurvivalRate{self.name}': rate})


    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame: # type: ignore[override]
        if y is None:
            raise ValueError('MyTargetEncoder requires y')
        y_arr = np.asarray(y, dtype=float)
        known = np.zeros(len(X), dtype=int)
        rate = np.zeros(len(X))
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_index, val_index in kfold.split(X):
            stats = self._stats(X.iloc[train_index], y_arr[train_index], list(self.group_cols))
            known[val_index], rate[val_index] = self._encode(
                X.iloc[val_index], stats, float(y_arr[train_index].mean())
            )
        self.fit(X, y_arr)
        return X.assign(**{f'IsGroup{self.name}': known, f'SurvivalRate{self.name}': rate})


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([self.feature_names_in_, [f'IsGroup{self.name}', f'SurvivalRate{self.name}']])
