import logging
import yaml
from typing import Optional, Sequence, Literal, DefaultDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torch_geometric.data import Data
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer
from pytorch_lightning.utilities import grad_norm

from omegaconf import DictConfig
from hydra.utils import instantiate

from .egnn import parameter_init
from .._tensor import assure_2d as _assure_2d

logger = logging.getLogger(__name__)


class StandardizeFeatures(nn.Module):
    """Standardize features in geometric data object."""

    def __init__(
        self,
        stats_path: str,
    ) -> None:
        """
        Arguments:
            stats_path: YAML file containing mean and standard deviation for
                features that should be standardized.
        """

        super().__init__()

        with open(stats_path, "r") as yf:
            feature_stats = yaml.safe_load(yf)

        self.stats = nn.ParameterDict(
            {
                name: nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.tensor(stats["mean"]),
                            requires_grad=False,
                        ),
                        nn.Parameter(
                            torch.tensor(stats["std"]),
                            requires_grad=False,
                        ),
                    ]
                )
                for name, stats in feature_stats.items()
            }
        )

    def forward(self, data: Data) -> Data:
        """Standardize features.

        Arguments:
            data: Geometric data object.

        Returns:
            Geometric data object with standardized features.
        """

        for name, (mean, std) in self.stats.items():
            x = getattr(data, name)
            x -= mean
            x /= std + 1e-9
            setattr(data, name, x)

        return data


class ResidueEmbedding(nn.Module):
    """Computes residue embeddings and adds them to the node features."""

    def __init__(
        self,
        num_embeddings: int = 20,
        embedding_dim: int = 12,
    ) -> None:
        """
        Arguments:
            num_embeddings: Number of embeddings.
            embedding_dim: Desired embedding dimension.
        """

        super().__init__()

        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, data: Data) -> Data:
        """Compute residue embeddings and add them to the node feautres.

        Arguments:
            data: Geometric data object.

        Returns:
            Geometric data object with residue embeddings added to the node
            features.
        """

        data.x = torch.cat(
            [
                self.embed(data.residue),
                _assure_2d(data.x),
            ],
            dim=-1,
        )

        return data


class ElementEmbedding(nn.Module):
    """Computes element embeddings and adds them to the node features."""

    def __init__(
        self,
        num_embeddings: int = 5,
        embedding_dim: int = 4,
    ) -> None:
        """
        Arguments:
            num_embeddings: Number of embeddings.
            embedding_dim: Desired embedding dimension.
        """

        super().__init__()

        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, data: Data) -> Data:
        """Compute element features and add them to the node features.

        Arguments:
            data: Geometric data object.

        Returns:
            Geometric data object with element embeddings added to the node
            features.
        """

        data.x = torch.cat(
            [
                self.embed(data.element),
                _assure_2d(data.x),
            ],
            dim=-1,
        )

        return data


class LinearTransformNode(nn.Module):
    """Linear transform node features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        """
        Arguments:
            in_features: Number of input node features.
            out_features: Number of output node features.
            bias: Whether bias should be added to linear transformation.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias,
        )

    def forward(self, data: Data) -> Data:
        """Linear transform node features.

        Arguments:
            data: Geometric data object.

        Returns:
            Geometric data object with linear transformed node features.
        """

        data.x = self.linear(data.x)

        return data


class LinearTransformEdge(nn.Module):
    """Linear transform edge features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        """
        Arguments:
            in_features: Number of input edge features.
            out_features: Number of output edge features.
            bias: Whether bias should be added to linear transformation.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias,
        )

    def forward(self, data: Data) -> Data:
        """Linear transform edge features.

        Arguments:
            data: Geometric data object.

        Returns:
            Geometric data object with linear transformed edge features.
        """

        data.edge_attr = self.linear(data.edge_attr)

        return data


class GeometricModule(LightningModule):
    """Generic lightning module for geometric models."""

    def __init__(
        self,
        tail: DictConfig,
        body: DictConfig,
        head: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig] = None,
        example_input: Optional[Tensor] = None,
        log_net_stats: Sequence[
            Literal[
                "grad_norm",
                "weight_stat",
                "pos_coef",
                "deq_abs",
                "deq_rel",
                "deq_nstep",
            ]
        ] = [],
    ) -> None:
        """
        Arguments:
            tail: Tail model configuration.
            body: Body model configuration.
            head: Head model configuration.
            loss: Loss configuration.
            metrics: Metrics configuration.
            optimizer: Optimizer configuration.
            scheduler: Learning-rate scheduler configuration.
            example_input: Example input.
            log_net_stats: Which network statistics to log:
                - `"grad_norms"`: Gradient norms.
                - `"weight_stats"`: Mean and standard deviation of weights.
                - `"pos_coef"`: Scale and exponent for positional updates.
        """

        super().__init__()

        # self.save_hyperparameters(
        #     ignore=[
        #         "example_input",
        #     ]
        # )

        self.tail = instantiate(tail)  # .to(device=self.device)
        self.body = instantiate(body)
        self.head = instantiate(head)

        self.loss = instantiate(loss)

        self.train_metrics = MetricCollection(
            instantiate(metrics, _convert_="all"),
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            instantiate(metrics, _convert_="all"),
            prefix="val/",
        )
        self.test_metrics = MetricCollection(
            instantiate(metrics, _convert_="all"),
            prefix="test/",
        )

        self.misc_metrics = DefaultDict(list)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.example_input_array = example_input
        self.log_net_stats = log_net_stats

        if hasattr(self.tail, "reset_parameters"):
            self.tail.reset_parameters()
        else:
            self.tail.apply(lambda module: parameter_init(module, "linear"))

        if hasattr(self.body, "reset_parameters"):
            self.body.reset_parameters()
        else:
            self.body.apply(parameter_init)

        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()
        else:
            self.head.apply(lambda module: parameter_init(module, "linear"))

    def _forward(
        self,
        data: Data,
    ) -> Tensor:
        """Compute forward pass.

        Arguments:
            data: Model input.

        Returns:
            Logits, positional coordinates, optional penalty and optional info
            dictionary containing possible metrics generated by sub-optimizers.
        """

        data = self.tail(data)

        x, pos, *rest = self.body(
            data.x,
            data.pos.to(data.x.dtype),
            data.edge_index,
            data.edge_attr.to(data.x.dtype),
            getattr(data, "batch", None),
        )

        y_hat = self.head(x)

        if 1 <= len(rest):
            penalty = rest[0]
        else:
            penalty = None

        if 2 <= len(rest):
            info = rest[1]
        else:
            info = None

        return y_hat, pos, penalty, info

    def forward(
        self,
        data: Data,
    ) -> Tensor:
        """Compute forward pass.

        Arguments:
            data: Model input.

        Returns:
            Logits.
        """

        y_hat, *_ = self._forward(data)

        return y_hat

    def compute(self, batch: Data) -> tuple[Tensor, Tensor]:
        """Compute probabilities.

        Arguments:
            batch: Data batch.

        Returns:
            Predicted probabilities, targets, optional penalty and info
            dictionary containing metrics generated by sub-optimizers.
        """

        # p = F.sigmoid(self.forward(batch)).squeeze()
        # y = batch.y.float()

        y_hat, _, penalty, info = self._forward(batch)

        p = F.sigmoid(y_hat).squeeze()
        y = batch.y.float()

        return p, y, penalty, info

    def training_step(
        self,
        batch: Data,
        batch_idx: int,
    ) -> Tensor:
        """Compute probabilities, loss and metrics.

        Arguments:
            batch: Data batch.
            batch_idx: Index of batch.

        Returns:
            Loss.
        """

        p, y, penalty, info = self.compute(batch)

        loss = self.loss(p, y)
        if penalty is not None:
            loss += penalty
        metrics = self.train_metrics(p, y)

        self.log("train/loss", loss, on_step=True)
        self.log_dict(metrics, on_step=True, prog_bar=True)

        if info is not None:
            if "deq_abs" in self.log_net_stats:
                self.log("deq/abs", info["abs_lowest"].mean())
            if "deq_rel" in self.log_net_stats:
                self.log("deq/rel", info["rel_lowest"].mean())
            if "deq_nstep" in self.log_net_stats:
                self.log("deq/nstep", info["nstep"].mean())

        return loss

    def validation_step(
        self,
        batch: Data,
        batch_idx: int,
    ) -> None:
        """Update validation metrics.

        Arguments:
            batch: Data batch.
            batch_idx: Index of batch.
        """

        p, y, _, info = self.compute(batch)

        self.val_metrics.update(p, y)

        if info is not None and "sradius" in info:
            self.misc_metrics["deq/sradius"].append(info["sradius"].max())

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""

        metrics = self.val_metrics.compute()

        self.log_dict(metrics, prog_bar=True)

        if "deq/sradius" in self.misc_metrics:
            sradi = self.misc_metrics["deq/sradius"]
            sradius = sum(sradi) / len(sradi)
            if "deq_sradius" in self.log_net_stats:
                self.log("deq/sradius", sradius)
            del self.misc_metrics["deq/sradius"]

    def test_step(
        self,
        batch: Data,
        batch_idx: int,
    ) -> None:
        """Update test metrics.

        Arguments:
            batch: Data batch.
            batch_idx: Index of batch.
        """

        p, y, *_ = self.compute(batch)

        self.test_metrics.update(p, y)

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics."""

        metrics = self.test_metrics.compute()

        self.log_dict(metrics)

    def configure_optimizers(self) -> tuple[list, list]:
        """Configure optimizer.

        Returns:
            Tuple with list containing one optimizer and a list that is either
            empty or that contains one learning-rate scheduler.
        """

        optimizers = [instantiate(self.optimizer, params=self.parameters())]
        schedulers = []

        if self.scheduler is not None:
            schedulers.append(instantiate(self.scheduler, optimizer=optimizers[0]))

        return optimizers, schedulers

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Log model statistics.

        Arguments:
            optimizer: Optimizer that is currently used.
        """

        if "grad_norm" in self.log_net_stats:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

        if "weight_stat" in self.log_net_stats:
            norms = {}
            for name, p in self.named_parameters():
                mean = p.data.mean()
                std = p.data.std()
                min = p.data.min()
                max = p.data.max()
                if not mean.isnan():
                    norms[f"weight_stat_mean/{name}"] = mean
                if not std.isnan():
                    norms[f"weight_stat_std/{name}"] = std
                if not min.isnan():
                    norms[f"weight_stat_min/{name}"] = min
                if not max.isnan():
                    norms[f"weight_stat_max/{name}"] = max
            self.log_dict(norms)

        if "pos_coef" in self.log_net_stats:
            norms = {}
            for name, p in self.named_parameters():
                if name.endswith("pos_scope") or name.endswith("pos_scale"):
                    norms[f"pos_coef/{name}"] = p.data
            self.log_dict(norms)
