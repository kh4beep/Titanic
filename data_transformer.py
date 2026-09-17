import os
import pickle

import pandas as pd


class DataTransformer:
    def __init__(self, *, config, path: str, model_name: str):
        self.config = config
        self.path = path
        self.model_name = model_name
        self.state_checkpoint = {}


    def fit(self, *, df: pd.DataFrame) -> None:
        pass


    def transform(self, *, df: pd.DataFrame) -> pd.DataFrame:
        return df


    def fit_transform(self, *, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df=df)
        return self.transform(df=df)


    def save_state(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), "wb") as f:
            pickle.dump(self.state_checkpoint, f)

        # Save each transformer model


    def load_state(self) -> None:
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'rb') as f:
            self.state_checkpoint = pickle.load(f)

        # Load each transformer model
