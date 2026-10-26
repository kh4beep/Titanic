import numpy as np
import pandas as pd

from omegaconf import DictConfig
from dataset_manager import DatasetManager
from model_manager import ModelManager

from utils import create_submission

def run(*, config: DictConfig) -> dict:
    """Load the dataset for the current mode, then train models or write a submission."""
    if config.general.is_train:
        df = pd.read_csv(config.paths.dataset, index_col=config.general.index_col)
    else:
        df = pd.read_csv(config.paths.test_dataset, index_col=config.general.index_col)

    dataset_manager = DatasetManager(config=config)
    df = dataset_manager.get_dataset(df=df)

    model_manager = ModelManager(config=config)
    if config.general.is_train:
        return model_manager.fit(df=df)

    return create_submission(
        index=np.array(df.index),
        predictions=model_manager.predict(X=df),
        path=config.paths.submission
    )
