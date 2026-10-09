import random
import os
import sys
import torch
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def set_seed(*, seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def setup_logger(*, path: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )

    file_handler = logging.FileHandler(path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_metrics(*, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_pred)
    y_label = (y_proba >= threshold).astype(int)
    result = {}
    result['accuracy'] = float(accuracy_score(y_true, y_label))
    result['f1'] = float(f1_score(y_true, y_label))
    result['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    return result


def create_submission(*, index: np.ndarray, predictions: np.ndarray, path: str) -> dict:
    artifacts = {}
    submission = pd.DataFrame({
        'PassengerId': index,
        'Survived': predictions
    })
    submission.to_csv(path, index=False)
    artifacts['shape'] = submission.shape
    artifacts['value_counts'] = submission['Survived'].value_counts()
    return artifacts


def create_metrics_tables(*, metrics: dict) -> str:
    rows = []
    for model, m in metrics.items():
        per_fold = pd.DataFrame(m['per_fold']) if 'per_fold' in m else pd.DataFrame()
        for name in m['oof']:
            rows.append({
                'model': model, 'metric': name,
                'oof_mean': per_fold[name].mean() if name in per_fold else np.nan,
                'oof_std': per_fold[name].std() if name in per_fold else np.nan,
                'oof': m['oof'][name],
                'test': m.get('test', {}).get(name, np.nan),
            })
    df = pd.DataFrame(rows).set_index(['metric', 'model']).astype(float).round(4)
    return '\n' + '\n\n'.join(f'{metric}:\n{df.loc[metric].to_string()}' for metric in df.index.unique('metric'))
