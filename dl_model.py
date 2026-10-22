import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class DLM(nn.Module):
    def __init__(
        self,
        *,
        output_size: int = 1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        warmup_epochs: int = 10,
        plateau_epochs: int = 10,
        min_lr_ratio: float = 0.05,
        device: str = 'cpu',
        early_stopping: bool = False,
        tol: float = 0.001,
        validation_fraction: float = 0.2,
        n_iter_no_change: int = 25,
        verbose: int = 0,
        random_state: int = 42,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epochs
        self.min_lr_ratio = min_lr_ratio
        self.device = device
        self.early_stopping = early_stopping
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state
        self.net: nn.Module | None = None
        self.input_size: int | None = None
        self.history_: dict[str, list] = {}
        self.best_epoch_: int | None = None
        self.stopped_epoch_: int | None = None


    def _build_net(self) -> None:
        if self.input_size is None:
            raise RuntimeError('input_size is not set')
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.BatchNorm1d(32),
            nn.Hardswish(),
            nn.Dropout(0.8),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Hardswish(),

            nn.Linear(16, self.output_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError('net is not built, call fit or load first')
        return self.net(x)


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DLM':
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self.input_size = X.shape[1]
        self._build_net()
        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_multiplier)
        loss_fn = nn.BCEWithLogitsLoss()

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                stratify=y,
                random_state=self.random_state,
            )
            X_val_t, y_val_t = self._to_tensors(X_val, y_val)
        else:
            X_val_t, y_val_t = None, None

        X_train_t, y_train_t = self._to_tensors(X, y)

        self.history_ = {'epoch': [], 'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'lr': []}
        best_loss = np.inf
        best_epoch = 0
        best_state = copy.deepcopy(self.state_dict())
        epochs_without_improvement = 0
        epoch = -1

        for epoch in range(self.epochs):
            self.history_['lr'].append(optimizer.param_groups[0]['lr'])
            self._train_epoch(X_train_t, y_train_t, optimizer=optimizer, loss_fn=loss_fn, rng=rng)
            scheduler.step()

            train_loss, train_auc = self._evaluate(X_train_t, y_train_t, loss_fn=loss_fn)
            self.history_['epoch'].append(epoch + 1)
            self.history_['train_loss'].append(train_loss)
            self.history_['train_auc'].append(train_auc)
            if self._should_log(epoch):
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.epochs}'
                    f'\t\tTrain loss: {train_loss:.4f}\tTrain ROC AUC: {train_auc:.4f}'
                )

            if X_val_t is None or y_val_t is None:
                continue

            val_loss, val_auc = self._evaluate(X_val_t, y_val_t, loss_fn=loss_fn)
            self.history_['val_loss'].append(val_loss)
            self.history_['val_auc'].append(val_auc)
            if self._should_log(epoch):
                self.logger.info(f'\t\t\tVal loss: {val_loss:.4f}\tVal ROC AUC: {val_auc:.4f}')

            if val_loss < best_loss - self.tol:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.n_iter_no_change:
                self.logger.info(
                    f'Early stopping on epoch {epoch + 1}, val loss {val_loss:.4f} | '
                    f'best epoch {best_epoch}, val loss {best_loss:.4f}'
                )
                break

        self.stopped_epoch_ = epoch + 1
        if self.early_stopping:
            self.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
        else:
            self.best_epoch_ = self.stopped_epoch_
        return self


    def get_training_artifacts(self) -> dict:
        if not self.history_:
            raise RuntimeError('Model is not fitted')
        return {
            'history': copy.deepcopy(self.history_),
            'best_epoch': self.best_epoch_,
            'stopped_epoch': self.stopped_epoch_,
        }


    def _lr_multiplier(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        decay_start = self.warmup_epochs + self.plateau_epochs
        decay_end = self.epochs - self.plateau_epochs
        if epoch < decay_start:
            return 1.0
        if epoch >= decay_end:
            return self.min_lr_ratio
        progress = (epoch - decay_start) / (decay_end - decay_start)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


    def _to_tensors(self, X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return X_t, y_t


    def _should_log(self, epoch: int) -> bool:
        return bool(self.verbose and epoch % self.verbose == 0) or epoch + 1 == self.epochs


    def _train_epoch(
        self, X_t: torch.Tensor, y_t: torch.Tensor, *, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, rng: np.random.Generator
    ) -> None:
        self.train()
        indices = rng.permutation(X_t.shape[0])
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if len(batch_indices) < 2:
                continue
            batch_X = X_t[batch_indices].to(self.device)
            batch_y = y_t[batch_indices].to(self.device)

            optimizer.zero_grad()
            loss = loss_fn(self.forward(batch_X), batch_y)
            loss.backward()
            optimizer.step()


    @torch.no_grad()
    def _predict(self, X_t: torch.Tensor) -> torch.Tensor:
        self.eval()
        outputs = [
            self.forward(X_t[start:start + self.batch_size].to(self.device)).cpu()
            for start in range(0, X_t.shape[0], self.batch_size)
        ]
        return torch.cat(outputs).ravel()


    def _evaluate(self, X_t: torch.Tensor, y_t: torch.Tensor, *, loss_fn: nn.Module) -> tuple[float, float]:
        logits = self._predict(X_t)
        y_true = y_t.ravel()
        loss = float(loss_fn(logits, y_true))
        auc = float(roc_auc_score(y_true.numpy(), logits.numpy()))
        return loss, auc


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = torch.sigmoid(self._predict(torch.tensor(X, dtype=torch.float32))).numpy()
        return np.column_stack([1 - preds, preds])


    def save(self, *, path: str) -> None:
        torch.save({
            'params': {
                'output_size': self.output_size,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'warmup_epochs': self.warmup_epochs,
                'plateau_epochs': self.plateau_epochs,
                'min_lr_ratio': self.min_lr_ratio,
                'device': self.device,
                'early_stopping': self.early_stopping,
                'tol': self.tol,
                'validation_fraction': self.validation_fraction,
                'n_iter_no_change': self.n_iter_no_change,
                'verbose': self.verbose,
                'random_state': self.random_state,
            },
            'input_size': self.input_size,
            'state_dict': self.state_dict(),
            'training_artifacts': self.get_training_artifacts() if self.history_ else None,
        }, path)


    @classmethod
    def load(cls, *, path: str) -> 'DLM':
        ckpt = torch.load(path, map_location='cpu')
        model = cls(**ckpt['params'])
        model.input_size = ckpt['input_size']
        model._build_net()
        model.load_state_dict(ckpt['state_dict'])
        model.to(model.device)
        artifacts = ckpt.get('training_artifacts')
        if artifacts:
            model.history_ = artifacts['history']
            model.best_epoch_ = artifacts['best_epoch']
            model.stopped_epoch_ = artifacts['stopped_epoch']
        return model
