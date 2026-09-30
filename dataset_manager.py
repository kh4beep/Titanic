import pickle

import pandas as pd

class DatasetManager:
    def __init__(self, *, config) -> None:
        self.config = config

        if config.general.is_train and config.general.is_rebuild_dataset:
            self.dataset_checkpoint = {}
        else:
            self._load_dataset_checkpoint()


    def get_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
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
        print('Initial shape:', df.shape)
        # def get_deck(cabin):
        #     if type(cabin) == str:
        #         parts = cabin.split()
        #         if len(parts) == 1:
        #             return cabin[0]
        #         for part in parts:
        #             if len(part) > 1:
        #                 return part[0]
        #     return 'Unknown'
        # df['Deck'] = df.Cabin.apply(get_deck)
        # df['Surname'] = df.Name.apply(lambda x: x[:x.find(',')])
        # df['Title'] = df.Name.apply(lambda x: x[x.find(',')+2:x.find('.')])

        if self.config.general.is_train:
            # Add dataset transformations here
            drop_columns = ['Name', 'Ticket', 'Cabin']
            self.dataset_checkpoint['dtypes'] = dict(df.dtypes)
            self.dataset_checkpoint['dropcols'] = drop_columns
        else:
            # Add dataset transformations here
            drop_columns = self.dataset_checkpoint['dropcols']

        df = df.drop(drop_columns, axis=1)
        
        print('Shape after build dataset:', df.shape)
        print('There are', df.isna().sum().sum(), 'missing values.')
        print('There are', (df.nunique() == 2).sum(), 'binary columns.')

        return df
