import os
import pickle
import logging

import pandas as pd
import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from data_transformer import DataTransformer
from model_trainer import ModelTrainer
from utils import get_metrics


class ModelManager:
    """
    Orchestrate CV training, OOF/hold-out evaluation and blended inference.
    For every model and fold keeps a (DataTransformer, ModelTrainer) pair; inference
    averages fold probabilities per model, then blends models with to_inference weights.
    """
    def __init__(self, *, config: DictConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.threshold = config.general.predict_threshold

        if config.general.is_train:
            self.models = {}
            self.state_checkpoint = {'models': {}}
        else:
            self._load_state()
            self._load_models()


    def fit(self, *, df: pd.DataFrame) -> dict:
        """
        Train all models from to_train with K-fold CV and evaluate them.
        Optionally holds out a test part first. Writes oof.csv and feature importances;
        returns artifacts with best params and OOF/test metrics.
        """
        X = df.drop('Survived', axis=1)
        y = df.Survived
        X_test, y_test = None, None
        if self.config.general.is_hold_test:
            X, X_test, y, y_test = train_test_split(
                X,
                y, 
                test_size=self.config.general.test_size, 
                random_state=self.config.general.seed,
                stratify=y
            )

        artifacts = {'best_params': {}}
        metrics = {}
        importances = {}
        shap_importances = {}
        all_oof_predictions = {}
        splitter = StratifiedKFold(
            n_splits=self.config.general.n_splits, 
            shuffle=self.config.general.is_shuffle, 
            random_state=self.config.general.seed if self.config.general.is_shuffle else None
        )
        folds = list(splitter.split(X=X, y=y))

        for k, v in self.config.models.to_inference.items():
            if v and not self.config.models.to_train.get(k):
                raise ValueError(f'Model {k} has weight {v} in to_inference, but is not in to_train')

        for model_name, is_train_model in self.config.models.to_train.items():
            if not is_train_model:
                continue
            all_oof_predictions[model_name] = self._train_one_model(
                X=X, y=y, model_name=model_name, folds=folds, artifacts=artifacts, metrics=metrics, importances=importances, shap_importances=shap_importances
            )

        columns = {}
        for (model_name, kind), per_fold in importances.items():
            all_folds = pd.concat(per_fold, axis=1)
            columns[f'{model_name}_{kind}_mean'] = all_folds.mean(axis=1)
            columns[f'{model_name}_{kind}_std'] = all_folds.std(axis=1)
        pd.DataFrame(columns).to_csv(self.config.paths.importance)

        if shap_importances:
            columns = {}
            for model_name, per_fold in shap_importances.items():
                folds = pd.concat(per_fold, axis=1)
                columns[f'{model_name}_mean'] = folds.mean(axis=1)
                columns[f'{model_name}_std'] = folds.std(axis=1)
            pd.DataFrame(columns).to_csv(self.config.paths.shap_importance)

        ensemble_oof_predictions = self._blend(all_predictions=all_oof_predictions)
        all_oof_predictions['ensemble'] = ensemble_oof_predictions
        metrics['ensemble'] = {'oof': get_metrics(y_true=np.asarray(y), y_proba=ensemble_oof_predictions, threshold=self.threshold)}
        pd.DataFrame(all_oof_predictions, index=X.index).assign(target=y).to_csv(self.config.paths.oof)

        if self.config.general.is_eval_test:
            if X_test is None or y_test is None:
                raise ValueError('X_test or y_test is not defined')
            
            all_predictions = self._predict_per_model(X=X_test)
            predictions = self._blend(all_predictions=all_predictions)

            for model_name, preds in all_predictions.items():
                metrics[model_name]['test'] = get_metrics(y_true=y_test, y_proba=preds, threshold=self.threshold)
                
            metrics['ensemble']['test'] = get_metrics(y_true=y_test, y_proba=predictions, threshold=self.threshold)
            artifacts['test_predictions'] = (predictions >= self.threshold).astype(int)

        artifacts['metrics'] = metrics
        return artifacts


    def predict(self, *, X: pd.DataFrame) -> np.ndarray:
        """Return class labels: blended probabilities thresholded at predict_threshold."""
        predictions = self._blend(all_predictions=self._predict_per_model(X=X))
        return (predictions >= self.threshold).astype(int)


    def _train_one_model(
            self, *, X: pd.DataFrame, y: pd.Series, model_name: str, folds: list, artifacts: dict, metrics: dict, importances: dict, shap_importances: dict
        ) -> np.ndarray:
        """
        Train one model on all folds and return its OOF probabilities.
        Hyperparameter search runs once, on the train part of fold 0; the found params
        are reused for every fold. Fills metrics, importances and artifacts in place.
        """
        self.logger.info(f'Start train {model_name}')
        model_checkpoint_path = os.path.join(self.config.paths.models, model_name)
        os.makedirs(model_checkpoint_path, exist_ok=True)

        self.models[model_name] = {}
        self.state_checkpoint['models'][model_name] = {}
        best_params = {}
        best_epochs = []
        shap_per_fold = []
        shap_data_per_fold = []
        metrics[model_name] = {'per_fold': []}
        oof = np.zeros(len(X))

        for i, (train_index, val_index) in tqdm(enumerate(folds), total=len(folds), desc=f'Model: {model_name}'):
            fold_checkpoint_path = os.path.join(model_checkpoint_path, f'fold_{i}')
            os.makedirs(fold_checkpoint_path, exist_ok=True)

            data_transformer_path = os.path.join(fold_checkpoint_path, 'transformers')
            data_transformer = DataTransformer(
                config=self.config, 
                path=data_transformer_path, 
                model_name=model_name,
                verbose=1 if i == 0 else 0,
            )

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]

            if self.config.general.is_refit_datatransformer:
                X_train = data_transformer.fit_transform(X=X_train, y=y_train)
            else:
                data_transformer.load_state()
                X_train = data_transformer.transform(X=X_train)

            X_val = data_transformer.transform(X=X.iloc[val_index])
            y_train = np.asarray(y_train)
            y_val = np.asarray(y.iloc[val_index])

            model_trainer_path = os.path.join(fold_checkpoint_path, 'model.pickle')

            if i == 0 and self.config.models.params[model_name].get('is_hyperparameter_search', False):
                best_params = ModelTrainer(
                    config=self.config,
                    path=model_trainer_path,
                    model_name=model_name
                ).get_best_params(X=X_train, y=y_train)
                artifacts['best_params'][model_name] = best_params

            model_trainer = ModelTrainer(
                config=self.config, 
                path=model_trainer_path, 
                model_name=model_name,
                best_params=best_params
            )

            artifacts_fold = model_trainer.fit(X=X_train, y=y_train)
            if 'history' in artifacts_fold:
                history = {k: v for k, v in artifacts_fold['history'].items() if v}  # без early_stopping списки val_* пустые
                pd.DataFrame(history).to_csv(os.path.join(fold_checkpoint_path, 'history.csv'), index=False)
                best_epochs.append(artifacts_fold['best_epoch'])
            val_predictions = model_trainer.predict_proba(X=X_val)
            val_metrics = get_metrics(y_true=y_val, y_proba=val_predictions, threshold=self.threshold)

            self.models[model_name][f'fold_{i}'] = {
                'data_transformer': data_transformer, 
                'model_trainer': model_trainer
            }
            self.state_checkpoint['models'][model_name][f'fold_{i}'] = {
                'data_transformer_path': data_transformer_path, 
                'model_trainer_path': model_trainer_path
            }

            metrics[model_name]['per_fold'].append(val_metrics)
            oof[val_index] = val_predictions

            names = data_transformer.get_feature_names()
            for kind, values in model_trainer.get_feature_importance().items():
                importances.setdefault((model_name, kind), []).append(pd.Series(values, index=names))

            if self.config.general.is_shap:
                shap_values = model_trainer.get_shap_values(X_background=X_train, X_explain=X_val)
                grouped = data_transformer.group_by_source_feature(shap_values)
                grouped.index = X.index[val_index]
                shap_per_fold.append(grouped)

                transformed_names = list(data_transformer.get_feature_names())
                transformed_values = pd.DataFrame(
                    np.asarray(X_val),
                    columns=transformed_names,
                    index=X.index[val_index],
                )
                raw_fold = X.iloc[val_index]
                shap_data = pd.DataFrame(index=X.index[val_index])

                for feature in grouped.columns:
                    if feature in raw_fold.columns:
                        shap_data[feature] = raw_fold[feature]
                    elif feature in transformed_values.columns:
                        shap_data[feature] = transformed_values[feature]
                    else:
                        shap_data[feature] = np.nan

                shap_data_per_fold.append(shap_data)

            if self.config.general.is_save_models:
                self._save_models(data_transformer=data_transformer, model_trainer=model_trainer)
                self._save_state()

        metrics[model_name]['oof'] = get_metrics(y_true=np.asarray(y), y_proba=oof, threshold=self.threshold)
        if best_epochs:
            self.logger.info(
                f'{model_name} best epoch by fold: {best_epochs}, '
                f'mean {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}'
            )
        if self.config.general.is_shap:
            oof_shap = pd.concat(shap_per_fold).loc[X.index]
            oof_shap_data = pd.concat(shap_data_per_fold).loc[X.index]
            oof_shap.to_csv(os.path.join(model_checkpoint_path, 'shap_oof.csv'))
            oof_shap_data.to_csv(os.path.join(model_checkpoint_path, 'shap_data_oof.csv'))
            shap_importances[model_name] = [fold.abs().mean() for fold in shap_per_fold]
            
        return oof


    def _predict_per_model(self, *, X: pd.DataFrame) -> dict:
        """Average fold probabilities for every loaded model."""
        all_predictions = {}
        for model_name, fold_models in self.models.items():
            all_predictions[model_name] = np.zeros(len(X))

            for models in fold_models.values():
                X_transformed = models['data_transformer'].transform(X=X)
                predictions = models['model_trainer'].predict_proba(X=X_transformed)
                all_predictions[model_name] += predictions / len(fold_models)

        return all_predictions


    def _blend(self, *, all_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of model probabilities; zero-weight models are ignored."""
        weights = {m: float(w) for m, w in self.config.models.to_inference.items() if w > 0}
        if not weights:
            raise ValueError('No models with positive weight in to_inference')

        blended = np.zeros_like(all_predictions[next(iter(weights))], dtype=float)
        for model_name, weight in weights.items():
            blended += all_predictions[model_name] * weight
        return blended / sum(weights.values())


    def _save_state(self) -> None:
        with open(self.config.paths.manager_checkpoint, "wb") as f:
            pickle.dump(self.state_checkpoint, f)


    def _load_state(self) -> None:
        with open(self.config.paths.manager_checkpoint, 'rb') as f:
            self.state_checkpoint = pickle.load(f)


    def _save_models(self, *, data_transformer: DataTransformer, model_trainer: ModelTrainer) -> None:
        data_transformer.save_state()
        model_trainer.save_state()


    def _load_models(self) -> None:
        self.models = {}
        for model_name, folds in self.state_checkpoint['models'].items():
            if not self.config.models.to_inference.get(model_name, False):
                continue

            self.models[model_name] = {}
            for fold_name, paths in tqdm(folds.items(), desc=f'Model: {model_name}'):
                data_transformer = DataTransformer(
                    config=self.config, 
                    path=paths['data_transformer_path'], 
                    model_name=model_name
                )
                data_transformer.load_state()

                model_trainer = ModelTrainer(
                    config=self.config, 
                    path=paths['model_trainer_path'], 
                    model_name=model_name
                )
                model_trainer.load_state()

                self.models[model_name][fold_name] = {
                    'data_transformer': data_transformer, 
                    'model_trainer': model_trainer
                }
