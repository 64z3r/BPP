import yaml
import warnings
from typing import TypeAlias, Literal, Sequence

from hydra import compose, initialize
from hydra.utils import instantiate

import torch
from torch import Tensor
from torch_geometric.data import Dataset

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


def compute_fwd_rev(conf):
    dataset_setup = instantiate(conf.data.module.dataset_setup)
    train_sets = instantiate(conf.data.module.train_sets)
    dataset = dataset_setup(train_sets)

    feature_stats = compute_stats(
        dataset,
        features=conf.data.features.standardize,
    )

    with open(conf.data.features.stats_path, "w") as yf:
        yaml.dump(feature_stats, yf)

    dm = instantiate(conf.data.module)
    dm.prepare_data()
    dm.setup("fit")

    dl = dm.train_dataloader()

    tail = instantiate(conf.model.tail).to(device="cuda")
    body = instantiate(conf.model.body).to(device="cuda")

    data = next(iter(dl))
    batch = tail(data.clone().to(device="cuda"))

    y = batch.x
    pos_a = pos_b = batch.pos
    for layer in body.layers:
        y, pos_a, pos_b = layer(
            y.clone(),
            pos_a.clone(),
            pos_b.clone(),
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
    for layer in reversed(body.layers):
        y, pos_a, pos_b = layer.reverse(
            y,
            pos_a,
            pos_b,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
    x = y

    return batch.x, x, batch.pos, pos_a, pos_b


torch.autograd.set_detect_anomaly(True)

torch.manual_seed(42)

with initialize(version_base=None, config_path="./src/bpp/conf/"):
    conf = compose(
        config_name="main",
        overrides=[
            "data=residue",
            "model=rev",
            "model.body.norm=true",
            "model.body.update=true",
            "model.body.layers=10",
            "model.body.disable=false",
            "model.body.dropout=0.5",
            "data.module.batch_size=4",
        ],
    )

x, x_hat, pos, pos_a_hat, pos_b_hat = compute_fwd_rev(conf)

d_x = x - x_hat
d_pos_a = pos - pos_a_hat
d_pos_b = pos - pos_b_hat

print(
    f"(x - x_hat)\n\t"
    f"mean={d_x.mean().item(): 13.6e} "
    f"std={d_x.std().item(): 13.6e} "
    f"min={d_x.min().item(): 13.6e} "
    f"max={d_x.max().item(): 13.6e}"
)
print(
    f"(pos - pos_a_hat)\n\t"
    f"mean={d_pos_a.mean().item(): 13.6e} "
    f"std={d_pos_a.std().item(): 13.6e} "
    f"min={d_pos_a.min().item(): 13.6e} "
    f"max={d_pos_a.max().item(): 13.6e}"
)

print(
    f"(pos - pos_b_hat)\n\t"
    f"mean={d_pos_b.mean().item(): 13.6e} "
    f"std={d_pos_b.std().item(): 13.6e} "
    f"min={d_pos_b.min().item(): 13.6e} "
    f"max={d_pos_b.max().item(): 13.6e}"
)
