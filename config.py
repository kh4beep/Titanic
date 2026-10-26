import os
from omegaconf import OmegaConf


config = {
    'general': {
        'experiment_name': 'ensemble',
        'seed': 42,
        'predict_threshold': 0.5,
        'is_train': True,
        'is_rebuild_dataset': True,
        'is_refit_datatransformer': True, # False reuses saved transformers; incompatible with target-derived features (MyTargetEncoder): train rows would lose cross-fitting
        'is_save_models': True,
        'is_hold_test': True,
        'is_eval_test': True,
        'test_size': 0.2,
        'n_splits': 5,
        'is_shuffle': True,
        'is_shap': True,
        'index_col': 'PassengerId'
    },
    'paths': {
        'dataset': '../../Datasets/titanic/train.csv',
        'test_dataset': '../../Datasets/titanic/test.csv',
        'checkpoints': './checkpoints/${general.experiment_name}',
        'logs': os.path.join('${paths.checkpoints}', 'logs.log'),
        'modified_dataset': os.path.join('${paths.checkpoints}', 'dataset.csv'),
        'dataset_checkpoint': os.path.join('${paths.checkpoints}', 'dataset_checkpoint.pickle'),
        'manager_checkpoint': os.path.join('${paths.checkpoints}', 'manager_checkpoint.pickle'),
        'models': os.path.join('${paths.checkpoints}', 'models'),
        'submission': os.path.join('${paths.checkpoints}', 'submission_${general.experiment_name}.csv'),
        'oof': os.path.join('${paths.checkpoints}', 'oof.csv'),
        'importance': os.path.join('${paths.checkpoints}', 'feature_importance.csv'),
        'shap_importance': os.path.join('${paths.checkpoints}', 'shap_importance.csv')
    },
    'models': {
        'params': {
            # Per-model config (key = sklearn class name or custom alias):
            #   is_calibration: bool              wrap in CalibratedClassifierCV(**calibration_kwargs)
            #   calibration_kwargs: dict
            #   is_hyperparameter_search: bool
            #   hyperparameter_search_kwargs:
            #       cv, n_iter, scoring
            #       param_distributions: {param: {'type': ..., ...}}  see ModelTrainer._make_distributions
            #   fit_kwargs: dict                  passed to fit(), not to the constructor (e.g. CatBoost cat_features)
            #   val_size: float                   CatBoost only: if > 0, this fraction of train is held out as eval_set
            #   kwargs: dict                      constructor params; also the prior for hyperparameter search,
            #                                     so every searched param must be set here explicitly
            'BaseLineLogisticRegression': 
            {
                'kwargs': {'random_state': '${general.seed}'}
            },
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
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 3,
                    'n_iter': 25,
                    'scoring': 'roc_auc',
                    'param_distributions': {
                        'learning_rate': {'type': 'loguniform', 'a': 0.01, 'b': 0.05},
                        'depth': {'type': 'int', 'a': 4, 'b': 8},
                        'l2_leaf_reg': {'type': 'loguniform', 'a': 1, 'b': 10},
                        'bagging_temperature': {'type': 'uniform', 'a': 0, 'b': 1}
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
                'kwargs': {
                    'output_size': 1,
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'warmup_epochs': 10,
                    'plateau_epochs': 10,
                    'min_lr_ratio': 0.05,
                    'device': 'cpu',
                    'early_stopping': False,
                    'tol': 0.001,
                    'validation_fraction': 0.2,
                    'n_iter_no_change': 25,
                    'verbose': 10,
                    'random_state': '${general.seed}'
                }
            },
        },
        'to_train': {
            'BaseLineLogisticRegression': True,
            'LinearSVC': False,
            'SVC': False,
            'LogisticRegression': False,
            'KNeighborsClassifier': True,
            'GaussianNB': False,
            'DecisionTreeClassifier': False,
            'RandomForestClassifier': False,
            'CatBoostClassifier': True,
            'DLM': True
        },
        'to_inference': {
            'BaseLineLogisticRegression': 0,
            'LinearSVC': 0,
            'SVC': 0,
            'LogisticRegression': 0,
            'KNeighborsClassifier': 0.5,
            'GaussianNB': 0,
            'DecisionTreeClassifier': 0,
            'RandomForestClassifier': 0,
            'CatBoostClassifier': 1,
            'DLM': 1,
        },
    }
}

config = OmegaConf.create(config)
