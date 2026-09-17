import pickle

import pandas as pd

class DatasetManager:
    def __init__(self, *, config) -> None:
        self.config = config

        if config.general.is_train and config.general.is_rebuild_dataset:
            self.dataset_checkpoint = {}
        else:
            self._load_dataset_checkpoint()


    def get_dataset(self, *, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if not self.config.general.is_train:
            df = self._build_dataset(df=df)

        elif self.config.general.is_rebuild_dataset:
            df = self._build_dataset(df=df)
            self._save_dataset(df=df)
            self._save_dataset_checkpoint()

        else:
            df = self._load_dataset()

        return df, self.dataset_checkpoint


    def _save_dataset_checkpoint(self) -> None:
        with open(self.config.paths.path_to_dataset_checkpoint, "wb") as f:
            pickle.dump(self.dataset_checkpoint, f)


    def _load_dataset_checkpoint(self) -> None:
        with open(self.config.paths.path_to_dataset_checkpoint, 'rb') as f:
            self.dataset_checkpoint = pickle.load(f)


    def _save_dataset(self, *, df: pd.DataFrame) -> None:
        df.to_csv(self.config.paths.path_to_modified_dataset, index=False)


    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.paths.path_to_modified_dataset)
        # Restore dataset state here
        df = df.astype(self.dataset_checkpoint['dtypes'])
        return df    

    
    def _build_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
        # Add dataset transformations here

        if self.config.general.is_train:
            # Add dataset transformations here
            self.dataset_checkpoint['dtypes'] = dict(df.dtypes)
        else:
            # Add dataset transformations here
            pass
        return df
