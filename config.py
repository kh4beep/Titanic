import os

from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'hyperparameterswoc',
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
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {'cv': 3},
                'kwargs': {'random_state': '${general.seed}'}
            },
            'SVC': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': True,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 25,
                    'scoring': 'roc_auc',
                    'param_distributions': {
                        'C': {'type': 'loguniform', 'a': 0.01, 'b': 100},
                        'kernel': {'type': 'choice', 'values': ['poly', 'rbf']},
                        'gamma': {'type': 'loguniform', 'a': 0.001, 'b': 1}}
                },
                'kwargs': {'C': 1.0, 'kernel': 'rbf', 'gamma': 0.03, 'random_state': '${general.seed}'}
            },
            'LogisticRegression': 
            {
                'is_calibration': False,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': True,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 25,
                    'scoring': 'roc_auc',
                    'param_distributions': {
                        'C': {'type': 'loguniform', 'a': 0.01, 'b': 10.0}
                    }
                },
                'kwargs': {'max_iter': 500, 'C': 1.0, 'random_state': '${general.seed}'}
            },
            'KNeighborsClassifier': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {'cv': 3},
                'kwargs': {'n_neighbors': 10, 'weights': 'distance'}
            },
            'GaussianNB': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {'cv': 3},
                'kwargs': {}
            },
            'DecisionTreeClassifier': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': True,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 100,
                    'scoring': 'roc_auc',
                    'param_distributions': {
                        'ccp_alpha': {'type': 'loguniform', 'a': 0.0001, 'b': 0.01},
                        'min_samples_leaf': {'type': 'logint', 'a': 1, 'b': 50},
                    }
                },
                'kwargs': {'ccp_alpha': 0.0001, 'min_samples_leaf': 3, 'random_state': '${general.seed}'}
            },
            'RandomForestClassifier': 
            {
                'is_calibration': True,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': True,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 25,
                    'scoring': 'roc_auc',
                    'n_jobs': -1,
                    'param_distributions': {
                        'max_features': {'type': 'uniform', 'a': 0.1, 'b': 0.8},
                        'min_samples_leaf': {'type': 'logint', 'a': 1, 'b': 30},
                        'max_samples': {'type': 'uniform', 'a': 0.4, 'b': 1.0}
                    }
                },
                'kwargs': {
                    'max_features': 0.25,
                    'min_samples_leaf': 7,
                    'max_samples': 0.9,
                    'random_state': '${general.seed}'
                }
            },
            'CatBoostClassifier': 
            {
                'is_calibration': False,
                'calibration_kwargs': {'cv': 3, 'ensemble': False},
                'is_hyperparameter_search': True,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 3,
                    'scoring': 'roc_auc',
                    'n_jobs': -1,
                    'param_distributions': {
                        'learning_rate': {'type': 'uniform', 'a': 0.01, 'b': 0.05},
                        # 'depth': {'type': 'int', 'a': 4, 'b': 8},
                        # 'l2_leaf_reg': {'type': 'loguniform', 'a': 1, 'b': 10},
                        # 'bagging_temperature': {'type': 'uniform', 'a': 0, 'b': 1}
                    }
                },
                'val_size': 0.0,
                'fit_kwargs': {'cat_features': ['Sex', 'Embarked', 'Title', 'Deck', 'Ticket', 'Surname']},
                'kwargs': {
                    'learning_rate': 0.03,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'bagging_temperature': 0.95,
                    'iterations': 300,
                    'one_hot_max_size': 3,
                    'verbose': 0, 
                    'allow_writing_files': False,
                    'random_state': '${general.seed}',
                }
            },
            'DLM': 
            {
                'kwargs': {'early_stopping': False, 'verbose': 10, 'random_state': '${general.seed}'}
            },
        },
        'to_train': {
            'LinearSVC': False,
            'SVC': True,
            'LogisticRegression': True,
            'KNeighborsClassifier': True,
            'GaussianNB': True,
            'DecisionTreeClassifier': True,
            'RandomForestClassifier': True,
            'CatBoostClassifier': True,
            'DLM': True
        },
        'to_inference': {
            'LinearSVC': 0,
            'SVC': 1,
            'LogisticRegression': 1,
            'KNeighborsClassifier': 1,
            'GaussianNB': 1,
            'DecisionTreeClassifier': 1,
            'RandomForestClassifier': 1,
            'CatBoostClassifier': 1,
            'DLM': 1,
        },
    }
}

config = OmegaConf.create(config)
