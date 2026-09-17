import os

from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'baseline',
        'seed': 42,
        'is_train': True,
        'is_rebuild_dataset': True,
        'is_refit_datatransformer': True,
        'is_save_models': True,
        'n_splits': 5,
        'is_shuffle': True,
        'target_column': None,
    },
    'paths': {
        'path_to_dataset': None,
        'path_to_checkpoints': './checkpoints/${general.experiment_name}',
        'path_to_config': os.path.join('${paths.path_to_checkpoints}', 'config.py'),
        'path_to_modified_dataset': os.path.join('${paths.path_to_checkpoints}', 'dataset.csv'),
        'path_to_dataset_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'dataset_checkpoint.pickle'),
        'path_to_manager_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'manager_checkpoint.pickle'),
        'path_to_models': os.path.join('${paths.path_to_checkpoints}', 'models'),
    },
    'models': {
        'params': {},
        'to_train': {},
        'to_inference': [],
    }
}

config = OmegaConf.create(config)
