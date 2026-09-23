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
        'is_hold_test': True,
        'is_eval_test': True,
        'test_size': 0.2,
        'n_splits': 5,
        'is_shuffle': True,
    },
    'paths': {
        'path_to_dataset': '../../Datasets/titanic/train.csv',
        'path_to_checkpoints': './checkpoints/${general.experiment_name}',
        'path_to_logs': os.path.join('${paths.path_to_checkpoints}', 'logs.log'),
        'path_to_config': os.path.join('${paths.path_to_checkpoints}', 'config.py'),
        'path_to_modified_dataset': os.path.join('${paths.path_to_checkpoints}', 'dataset.csv'),
        'path_to_dataset_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'dataset_checkpoint.pickle'),
        'path_to_manager_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'manager_checkpoint.pickle'),
        'path_to_models': os.path.join('${paths.path_to_checkpoints}', 'models'),
    },
    'models': {
        'params': {},
        'to_train': {
            'LogisticRegression': True,
            'DecisionTreeClassifier': True,
            'RandomForestClassifier': True,
        },
        'to_inference': {
            'LogisticRegression': 0.33,
            'DecisionTreeClassifier': 0.33,
            'RandomForestClassifier': 0.34,
        },
    }
}

config = OmegaConf.create(config)
