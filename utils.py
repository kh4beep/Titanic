import random
import os
import sys
import torch
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score


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


def get_metric(*, y_true: pd.Series, y_pred: pd.Series) -> float:
    return f1_score(y_true, np.round(y_pred))
