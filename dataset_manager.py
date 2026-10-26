import pickle
import logging

import pandas as pd

from omegaconf import DictConfig

class DatasetManager:
    """
    Build row-wise features and cache the resulting dataset.
    Only transformations that use a single row belong here
    """
    def __init__(self, *, config: DictConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config

        if config.general.is_train and config.general.is_rebuild_dataset:
            self.dataset_checkpoint = {}
        else:
            self._load_dataset_checkpoint()


    def get_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature dataset: rebuilt or loaded from cache depending on config."""
        if not self.config.general.is_train:
            df = self._build_dataset(df=df)
        elif self.config.general.is_rebuild_dataset:
            df = self._build_dataset(df=df)
            self._save_dataset(df=df)
            self._save_dataset_checkpoint()
        else:
            df = self._load_dataset()

        return df


    def _save_dataset_checkpoint(self) -> None:
        with open(self.config.paths.dataset_checkpoint, "wb") as f:
            pickle.dump(self.dataset_checkpoint, f)


    def _load_dataset_checkpoint(self) -> None:
        with open(self.config.paths.dataset_checkpoint, 'rb') as f:
            self.dataset_checkpoint = pickle.load(f)


    def _save_dataset(self, *, df: pd.DataFrame) -> None:
        df.to_csv(self.config.paths.modified_dataset, index=True)


    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.paths.modified_dataset, index_col=self.config.general.index_col)
        df = df.astype(self.dataset_checkpoint['dtypes'])
        return df    

    
    def _build_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.logger.info(f'Initial shape: {df.shape}')
        def get_deck(cabin):
            if isinstance(cabin, str):
                parts = cabin.split()
                if len(parts) == 1:
                    return cabin[0]
                for part in parts:
                    if len(part) > 1:
                        return part[0]
            return 'Unknown'
        
        df['Deck'] = df['Cabin'].apply(get_deck)
        df['Surname'] = df['Name'].apply(lambda x: x[:x.find(',')])
        df['Title'] = df['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])
        df.loc[~df.Title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Rare'

        if self.config.general.is_train:
            drop_columns = ['Name', 'Cabin']
            self.dataset_checkpoint['dropcols'] = drop_columns
        else:
            drop_columns = self.dataset_checkpoint['dropcols']

        df = df.drop(drop_columns, axis=1)
        self.dataset_checkpoint['dtypes'] = dict(df.dtypes)
        
        self.logger.info(f'Shape after build dataset: {df.shape}')
        self.logger.info(f'There are {df.isna().sum().sum()} missing values.')
        self.logger.info(f'There are {(df.nunique() == 2).sum()} binary columns.')

        return df
