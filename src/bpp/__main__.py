import logging
import yaml
import torch
import hydra
import wandb
import warnings

from typing import TypeAlias, Literal, Sequence
from torch import Tensor
from torch_geometric.data import Dataset
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from .model import GeometricModule

logger = logging.getLogger(__name__)

Stats: TypeAlias = dict[str, dict[Literal["mean", "std"], Tensor]]


def compute_stats(dataset: Dataset, features: Sequence[str]) -> Stats:
    """Computes mean and standard deviation of selected features in dataset.

    Arguments:
        dataset: Geometric dataset.
        features: Feature names that statistics should be computed for.

    Returns:
        Mean and standard deviations of selected features.
    """

    stats = {}

    for name in features:
        x = torch.cat([getattr(data, name) for data in dataset])
        stats[name] = {
            "mean": x.mean(axis=0).detach().cpu().tolist(),
            "std": x.std(axis=0).detach().cpu().tolist(),
        }

    return stats


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig) -> float | None:
    """Train and test model.

    Arguments:
        cfg: Configuration of train and test run.

    Returns:
        Metric of best performing checkpoint. The same metric is used as for
        determining the best checkpoint, but instead of being computed on the
        validation set it is computed on the test set.
    """

    try:
        project = cfg.logger.wandb.project
    except AttributeError:
        logger.warning("WandB logger not set, cannot determine project name.")
        project = None

    wandb.init(
        project=project,
        config=OmegaConf.to_container(cfg, resolve="all"),
        settings=wandb.Settings(start_method="thread"),
    )

    dataset_setup = instantiate(cfg.data.module.dataset_setup)
    train_sets = instantiate(cfg.data.module.train_sets)
    dataset = dataset_setup(train_sets)

    feature_stats = compute_stats(
        dataset,
        features=cfg.data.features.standardize,
    )

    with open(cfg.data.features.stats_path, "w") as yf:
        yaml.dump(feature_stats, yf)

    example_input = None

    for idx in range(len(dataset)):
        info = dataset.info(idx)
        if info["name"] == "2kug_1":
            example_input = dataset[idx]

    seed_everything(cfg.seed)

    model = GeometricModule(
        tail=cfg.model.tail,
        body=cfg.model.body,
        head=cfg.model.head,
        loss=cfg.loss,
        metrics=cfg.metrics,
        optimizer=cfg.optimizer,
        example_input=example_input,
        log_net_stats=cfg.model.log_net_stats,
    )

    print(model.body)

    datamodule = instantiate(cfg.data.module)

    trainer = instantiate(
        cfg.trainer,
        logger=list(instantiate(cfg.logger, _convert_="all").values()),
        callbacks=list(instantiate(cfg.callbacks, _convert_="all").values()),
    )

    with warnings.catch_warnings(action="ignore"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.model.checkpoint)
        metrics = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()

    try:
        metric = cfg.callbacks.checkpoint.monitor
        return metrics[-1][metric.replace("val", "test")]
    except AttributeError:
        logger.warning(
            "Checkpoints not tracked, cannot determine best test metric.",
        )
    except KeyError as error:
        logger.warning(
            f"Test metric {error} not found.",
        )


if __name__ == "__main__":
    main()
