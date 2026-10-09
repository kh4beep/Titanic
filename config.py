import os

from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'preprocessing42',
        'seed': 42,
        'predict_threshold': 0.5,
        'is_train': True,
        'is_rebuild_dataset': True,
        'is_refit_datatransformer': True, # False only without TargetEncoder and same
        'is_save_models': True,
        'is_hold_test': True,
        'is_eval_test': False,
        'test_size': 0.2,
        'n_splits': 5,
        'is_shuffle': True,
    },
    'paths': {
        'path_to_dataset': '../../Datasets/titanic/train.csv',
        'path_to_test_dataset': '../../Datasets/titanic/test.csv',
        'path_to_checkpoints': './checkpoints/${general.experiment_name}',
        'path_to_logs': os.path.join('${paths.path_to_checkpoints}', 'logs.log'),
        'path_to_config': os.path.join('${paths.path_to_checkpoints}', 'config.py'),
        'path_to_modified_dataset': os.path.join('${paths.path_to_checkpoints}', 'dataset.csv'),
        'path_to_dataset_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'dataset_checkpoint.pickle'),
        'path_to_manager_checkpoint': os.path.join('${paths.path_to_checkpoints}', 'manager_checkpoint.pickle'),
        'path_to_models': os.path.join('${paths.path_to_checkpoints}', 'models'),
        'path_to_submission': os.path.join('${paths.path_to_checkpoints}', 'submission.csv'),
        'path_to_oof': os.path.join('${paths.path_to_checkpoints}', 'oof.csv'),
        'path_to_importance': os.path.join('${paths.path_to_checkpoints}', 'feature_importance.csv')
    },
    'models': {
        'params': {
            'LinearSVC': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'method': 'sigmoid', 'cv': 3},
                'kwargs': {'random_state': '${general.seed}'}
            },
            'LogisticRegression': 
            {
                'is_calibration': False,
                'kwargs': {'random_state': '${general.seed}'}
            },
            'DecisionTreeClassifier': 
            {
                'is_calibration': False,
                'kwargs': {'random_state': '${general.seed}'}
            },
            'RandomForestClassifier': 
            {
                'is_calibration': False,
                'kwargs': {'random_state': '${general.seed}'}
            },
            'CatBoostClassifier': 
            {
                'is_calibration': False,
                'kwargs': {
                    'random_state': '${general.seed}',
                    'verbose': 100, 
                    'allow_writing_files': False
                }
            },
            'DLM': 
            {
                'is_calibration': False,
                'kwargs': {'verbose': 10, 'random_state': '${general.seed}'}
            },
        },
        'to_train': {
            'LinearSVC': True,
            'LogisticRegression': True,
            'DecisionTreeClassifier': True,
            'RandomForestClassifier': True,
            'CatBoostClassifier': True,
            'DLM': True
        },
        'to_inference': {
            'LinearSVC': 1,
            'LogisticRegression': 1,
            'DecisionTreeClassifier': 1,
            'RandomForestClassifier': 1,
            'CatBoostClassifier': 1,
            'DLM': 1,
        },
    }
}

config = OmegaConf.create(config)
