import os
import pickle

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from data_transformer import DataTransformer
from model_trainer import ModelTrainer
from utils import get_metric

class ModelManager:
    def __init__(self, *, config):
        self.config = config

        if config.general.is_train:
            self.models = {}
            self.state_checkpoint = {'models': {}}
        else:
            self._load_state()
            self._load_models()


    def fit(self, df: pd.DataFrame) -> dict:
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

        artifacts = {}
        splitter = StratifiedKFold(
            n_splits=self.config.general.n_splits, 
            shuffle=self.config.general.is_shuffle, 
            random_state=self.config.general.seed
        )
        folds = list(splitter.split(X=X, y=y))

        for model_name, is_train_model in self.config.models.to_train.items():
            if not is_train_model:
                continue

            model_checkpoint_path = os.path.join(self.config.paths.path_to_models, model_name)
            os.makedirs(model_checkpoint_path, exist_ok=True)

            self.models[model_name] = {}
            self.state_checkpoint['models'][model_name] = {}

            artifacts[model_name] = {'fold_metrics': []}
            oof = np.zeros(len(X))

            for i, (train_index, val_index) in tqdm(enumerate(folds), desc=f'Model: {model_name}'):
                fold_checkpoint_path = os.path.join(model_checkpoint_path, f'fold_{i}')
                os.makedirs(fold_checkpoint_path, exist_ok=True)

                data_transformer_path = os.path.join(fold_checkpoint_path, 'transformers')
                data_transformer = DataTransformer(
                    config=self.config, 
                    path=data_transformer_path, 
                    model_name=model_name
                )

                if self.config.general.is_refit_datatransformer:
                    X_train = data_transformer.fit_transform(X=X.iloc[train_index])
                else:
                    data_transformer.load_state()
                    X_train = data_transformer.transform(X=X.iloc[train_index])

                X_val = data_transformer.transform(X=X.iloc[val_index])
                y_train = np.asarray(y.iloc[train_index])
                y_val = np.asarray(y.iloc[val_index])

                model_trainer_path = os.path.join(fold_checkpoint_path, 'model.pickle')
                model_trainer = ModelTrainer(
                    config=self.config, 
                    path=model_trainer_path, 
                    model_name=model_name
                )

                artifacts_fold = model_trainer.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
                val_predictions = model_trainer.predict(X=X_val)
                val_metric = get_metric(y_true=y_val, y_pred=val_predictions)

                self.models[model_name][f'fold_{i}'] = {
                    'data_transformer': data_transformer, 
                    'model_trainer': model_trainer
                }
                self.state_checkpoint['models'][model_name][f'fold_{i}'] = {
                    'data_transformer_path': data_transformer_path, 
                    'model_trainer_path': model_trainer_path
                }

                artifacts[model_name]['fold_metrics'].append(val_metric)
                oof[val_index] = val_predictions

                if self.config.general.is_save_models:
                    self._save_models(data_transformer=data_transformer, model_trainer=model_trainer)
                    self._save_state()

            artifacts[model_name]['mean_metric'] = np.mean(artifacts[model_name]['fold_metrics'])
            artifacts[model_name]['oof_predictions'] = oof

        if self.config.general.is_eval_test:
            if X_test is None or y_test is None:
                raise Exception('X_test or y_test is not defined')
            test_predictions, predictions = self.predict(X=X_test)
            artifacts['test_metrics'] = {}
            for model_name, preds in test_predictions.items():
                artifacts['test_metrics'][model_name] = get_metric(y_true=y_test, y_pred=preds)
            artifacts['test_metrics']['ensemble'] = get_metric(y_true=y_test, y_pred=predictions)
            artifacts['test_predictions'] = predictions
        return artifacts


    def predict(self, *, X: pd.DataFrame) -> tuple[dict, np.ndarray] | np.ndarray:
        all_predictions = {}
        for model_name, folds in self.models.items():
            all_predictions[model_name] = np.zeros(len(X))
            for fold_name, models in folds.items():
                X_transformed = models['data_transformer'].transform(X=X)
                preds = np.asarray(models['model_trainer'].predict(X=X_transformed))
                all_predictions[model_name] += preds / len(folds)
            all_predictions[model_name] = np.clip(all_predictions[model_name], 0, 1)

        predictions = np.zeros(len(X))
        for model_name, weight in self.config.models.to_inference.items():
            predictions += all_predictions[model_name] * weight
        predictions = np.round(predictions).astype(int)
        return all_predictions, predictions


    def _save_state(self) -> None:
        with open(self.config.paths.path_to_manager_checkpoint, "wb") as f:
            pickle.dump(self.state_checkpoint, f)


    def _load_state(self) -> None:
        with open(self.config.paths.path_to_manager_checkpoint, 'rb') as f:
            self.state_checkpoint = pickle.load(f)


    def _save_models(self, *, data_transformer: DataTransformer, model_trainer: ModelTrainer) -> None:
        data_transformer.save_state()
        model_trainer.save_state()


    def _load_models(self) -> None:
        self.models = {}
        for model_name, folds in self.state_checkpoint['models'].items():
            if model_name not in self.config.models.to_inference.keys():
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
