import pandas as pd

from dataset_manager import DatasetManager
from model_manager import ModelManager
from submission import create_submission

def run(*, config) -> dict:
    df = pd.read_csv(config.paths.path_to_dataset)

    dataset_manager = DatasetManager(config=config)
    df = dataset_manager.get_dataset(df=df)

    model_manager = ModelManager(config=config)
    if config.general.is_train:
        return model_manager.fit(df=df)
    else:
        return create_submission(
            model_manager.predict(X=df)
        )
