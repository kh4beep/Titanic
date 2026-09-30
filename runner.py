import numpy as np
import pandas as pd

from dataset_manager import DatasetManager
from model_manager import ModelManager
from utils import create_submission

def run(*, config) -> dict:
    if config.general.is_train:
        df = pd.read_csv(config.paths.path_to_dataset, index_col='PassengerId')
    else:
        df = pd.read_csv(config.paths.path_to_test_dataset, index_col='PassengerId')

    dataset_manager = DatasetManager(config=config)
    df = dataset_manager.get_dataset(df=df)

    model_manager = ModelManager(config=config)
    if config.general.is_train:
        return model_manager.fit(df=df)
    else:
        _, predictions = model_manager.predict(X=df)
        return create_submission(
            index=np.array(df.index),
            predictions=predictions,
            path=config.paths.path_to_submission
        )
