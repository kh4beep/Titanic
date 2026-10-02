import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.optim import Adam
from utils import get_metric


class DLM(nn.Module):
    def __init__(
        self, 
        *, 
        output_size: int = 1, 
        epochs: int = 25, 
        batch_size: int = 32, 
        learning_rate: float = 0.001,
        device: str = 'cpu',
        random_state: int = 42,
    ):
        super(DLM, self).__init__()
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.random_state = random_state
        self.mlp = None
        self.input_size = None


    def _build_model(self):
        if self.input_size is None:
            raise Exception('input_size is None')
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            nn.Sigmoid()
        )


    def forward(self, X: torch.Tensor):
        if self.mlp is None:
            raise Exception('mlp is not defined')
        return self.mlp(X)


    def fit(self, *, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        rng = np.random.default_rng(self.random_state)  
        self.input_size = X_train.shape[1]
        if self.mlp is None:
            self._build_model()
            self.to(self.device)
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCELoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        indexes = np.arange(X_train_t.shape[0])

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')

            self.train()
            train_loss = 0.0

            rng.shuffle(indexes)
            for start_index in range(0, len(indexes), self.batch_size):
                batch_indexes = indexes[start_index:start_index + self.batch_size]
                batch_X = X_train_t[batch_indexes].to(self.device)
                batch_y = y_train_t[batch_indexes].to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print('Training loss:', train_loss / np.ceil(X_train_t.shape[0] / self.batch_size))

            self.eval()
            val_loss = 0.0
            predictions = []

            with torch.no_grad():
                for start_index in range(0, X_val_t.shape[0], self.batch_size):
                    batch_X = X_val_t[start_index:start_index + self.batch_size].to(self.device)
                    batch_y = y_val_t[start_index:start_index + self.batch_size].to(self.device)

                    outputs = self.forward(batch_X)
                    loss = loss_fn(outputs, batch_y)

                    val_loss += loss.item()
                    predictions.append(outputs.cpu().numpy())

            predictions = np.squeeze(np.concatenate(predictions))
            val_f1 = get_metric(y_true=y_val, y_pred=(predictions >= 0.5).astype(int))
            print('Validation loss:', val_loss / np.ceil(X_val_t.shape[0] / self.batch_size), '\nValidation F1:', val_f1)


    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            predictions = []
            for start_index in range(0, X_t.shape[0], self.batch_size):
                batch_X = X_t[start_index:start_index + self.batch_size].to(self.device)

                outputs = self.forward(batch_X)
                predictions.append(outputs.cpu().numpy())

            preds = np.squeeze(np.concatenate(predictions))
            return np.column_stack([1 - preds, preds])


    def save(self, *, path: str) -> None:
        torch.save({
            'params': {
                'output_size': self.output_size,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'device': self.device,
                'random_state': self.random_state
            },
            'input_size': self.input_size,
            'state_dict': self.state_dict(),
        }, path)


    @classmethod
    def load(cls, *, path: str) -> 'DLM':
        ckpt = torch.load(path, map_location='cpu')
        model = cls(**ckpt['params'])
        model.input_size = ckpt['input_size']
        model._build_model()
        model.load_state_dict(ckpt['state_dict'])
        model.to(model.device)
        return model
