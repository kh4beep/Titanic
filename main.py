import os
import time
import shutil

from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm

from config import config
from utils import set_seed, setup_logger, create_metrics_tables
from runner import run


CONFIG_SOURCE = Path(__file__).resolve().parent / 'config.py'


def save_config_snapshot(*, cfg: DictConfig, run_id: str) -> None:
    """
    Save the config of this run next to its checkpoints.
    Two files per run, never overwritten: the source config.py and a resolved YAML.
    """
    checkpoints = Path(cfg.paths.checkpoints)
    shutil.copyfile(CONFIG_SOURCE, checkpoints / f'config_{run_id}.py')
    OmegaConf.save(cfg, checkpoints / f'config_{run_id}.yaml', resolve=True)


def main(*, cfg: DictConfig) -> None:
    """Run one experiment: train/evaluate or build a submission, depending on cfg.general.is_train."""
    set_seed(seed=cfg.general.seed)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if cfg.general.is_train:
        os.makedirs(cfg.paths.checkpoints, exist_ok=True)
        os.makedirs(cfg.paths.models, exist_ok=True)

    logger = setup_logger(path=cfg.paths.logs)
    logger.info(f'Experiment {cfg.general.experiment_name}, run {run_id}')

    if cfg.general.is_train:
        save_config_snapshot(cfg=cfg, run_id=run_id)

    start_time = time.time()
    with logging_redirect_tqdm():
        artifacts = run(config=cfg)

    metrics = artifacts.pop('metrics', None)
    if artifacts:
        logger.info({k: v for k, v in artifacts.items()})
    if metrics is not None:
        logger.info(create_metrics_tables(metrics=metrics))
    logger.info(f'Total Time: {int(time.time() - start_time)} s')


if __name__ == '__main__':
    main(cfg=config)
