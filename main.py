import os
import time
import shutil
import logging

from tqdm.contrib.logging import logging_redirect_tqdm
from config import config
from utils import set_seed, setup_logger, create_metrics_tables
from runner import run


def main(*, config):
    set_seed(seed=config.general.seed)

    if config.general.is_train:
        os.makedirs(config.paths.path_to_checkpoints, exist_ok=True)
        os.makedirs(config.paths.path_to_models, exist_ok=True)
        shutil.copyfile('./config.py', config.paths.path_to_config)

    logger = setup_logger(path=config.paths.path_to_logs)

    logger.info(f'Experiment {config.general.experiment_name}')

    start_time = time.time()

    with logging_redirect_tqdm():
        artifacts = run(config=config)

    if 'metrics' in artifacts:
        rest = {k: v for k, v in artifacts.items() if k != 'metrics'}
        if rest:
            logger.info(rest)
        logger.info(create_metrics_tables(metrics=artifacts['metrics']))
    else:
        logger.info(artifacts)
    logger.info(f'Total Time: {int(time.time() - start_time)} s')


if __name__ == '__main__':
    main(config=config)