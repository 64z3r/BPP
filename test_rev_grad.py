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


def compute_grads(conf):
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
    body = instantiate(conf.model.body).to(
        device="cuda"
    )
    head = instantiate(conf.model.head).to(device="cuda")

    loss = instantiate(conf.loss)

    data = next(iter(dl))
    batch = tail(data.clone().to(device="cuda"))

    y = head(
        body(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)[0]
    )
    loss = loss(torch.nn.functional.sigmoid(y), batch.y.float())

    loss.backward()

    grads = [p.grad.clone() if p.grad is not None else None for p in body.parameters()]

    return grads

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

grads_1 = compute_grads(conf)

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
            "model.body.disable=true",
            "model.body.dropout=0.5",
            "data.module.batch_size=4",
        ],
    )

grads_2 = compute_grads(conf)

for i, (g1, g2) in enumerate(zip(grads_1, grads_2)):

    if g1 is None:
        if g2 is None:
            continue
        if (g2 == 0).all():
            print(f"({i:3d}) g1 is none and g2 is all zeros (size={g1.size()})")
            continue
        print(f"({i:3d}) g1 is none, but g2 isn't! (size={g1.size()})")
        continue

    if g2 is None:
        if g1 is None:
            continue
        if (g1 == 0).all():
            print(f"({i:3d}) g2 is none and g1 is all zeros (size={g1.size()})")
            continue
        print(f"({i:3d}) g2 is none, but g1 isn't! (size={g1.size()})")
        continue

    if g1.size() != g2.size():
        print(f"({i:d}) g1.size={g1.size()}, but g2.size={g2.size()}")

    d = g1 - g2
    with warnings.catch_warnings(action="ignore"):
        print(
            f"({i:3d}) "
            f"mean={d.mean().item(): 13.6e} "
            f"std={d.std().item(): 13.6e} "
            f"min={d.min().item(): 13.6e} "
            f"max={d.max().item(): 13.6e}"
        )
