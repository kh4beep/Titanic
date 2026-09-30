import os
import time
import shutil

from config import config
from utils import set_seed, setup_logger
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

    artifacts = run(config=config)
    
    logger.info(artifacts)    
    logger.info(f'Total Time: {int(time.time() - start_time)} s')


if __name__ == '__main__':
    main(config=config)