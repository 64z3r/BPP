import shutil
import pickle
import torch

from typing import Any, Callable, Literal, Sequence
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from joblib import delayed, cpu_count
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from .graph import Configuration, Constructor
from ._digest import digest
from ._parallel import ProgressParallel
from ._warnings import disable_warnings


class PUResNet(Dataset):
    """Dataset module for the PUResNet dataset.

    This module will download all necessary datasets and process them according
    to a provided configuration. You can also select which datasets should be
    used for training, validation or testing. The resulting samples are PyTorch
    Geometric data instances.

    References:
        .. [1] Kandel, J., Tayara, H. & Chong, K.T. PUResNet: prediction of
           protein-ligand binding sites using deep residual neural network. J
           Cheminform 13, 65 (2021). https://doi.org/10.1186/s13321-021-00547-7V
    """

    DATASET_URLS = {
        "scpdb": "https://github.com/jivankandel/PUResNet/raw/main/scpdb_subset.zip?download=",
        "coach": "https://github.com/jivankandel/PUResNet/raw/main/coach.zip?download=",
        "bu48": "https://github.com/jivankandel/PUResNet/raw/main/BU48.zip?download=",
    }

    DATASET_FILES = {
        "scpdb": "scpdb_subset.zip",
        "coach": "coach.zip",
        "bu48": "BU48.zip",
    }

    def __init__(
        self,
        sets: Sequence[Literal["scpdb", "coach", "bu48"]] = ["scpdb"],
        root: str = "./dataset/",
        conf: Configuration = Configuration(),
        transform: Callable[[Data], Data] | None = None,
        n_jobs: int = -1,
        verbose: Literal[0, 1, 2] = 1,
        cleanup_on_error: bool = True,
        constructor_cls: type[Constructor] = Constructor,
    ) -> None:
        """
        Arguments:
            sets: Datasets that should be included. Possible choices are:
                - `"scpdb"`: Subset of the scPDB dataset. *Hint*: use as
                    training set.
                - `"coach"`: Coach dataset with proteins removed that are
                    present in the scPDB dataset. *Hint*: use as validation set.
                - `"bu48"`: BU48 dataset with proteins removed that are present
                    in the scPDB and Coach datasets. *Hint*: add to validation
                    set or use as test set.
            root: Directory where to store the raw and processed datasets and
                samples.
            conf: Configuration that determines how the protein-ligand samples
                should be processed.
            transform: Callable that takes a data instance and returns a
                transformed version.
            n_jobs: Number of parallel jobs for processing the samples. If
                smaller or equal to zero, the maximum number of supported
                parallel threads will be used.
            verbose: Verbosity level. Choices are:
                - `1`: Nothing will be printed, except for some most critical
                    errors.
                - `2`: Only a progress bar will be shown.
                - `3`: All warnings etc. will be printed, without a progress
                    bar.
            cleanup_on_error: Whether the directory with processed samples
                should be deleted if an error occurs.
            constructor_cls: Class instance of the constructor for processing
                protein-ligand samples.
        """

        if n_jobs < 1:
            n_jobs = cpu_count()

        self.sets = sets
        self.conf = conf
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cleanup_on_error = cleanup_on_error
        self.constructor_cls = constructor_cls

        super().__init__(root, transform, log=(verbose > 0))

        self._processed = []
        for p in self.processed_file_names:
            p = Path(self.processed_dir) / p
            self._processed.extend(p.glob("*.pt"))
        self._processed.sort()

    @property
    def raw_file_names(self) -> list[str]:
        """File names of the raw datasets."""

        return [self.DATASET_FILES[name] for name in sorted(self.sets)]

    @property
    def processed_file_names(self) -> list[str]:
        """File names of processed samples. Actually, just the directories
        that contain the samples of a respective dataset.
        """

        names = []

        for name in sorted(self.sets):
            hash = digest(
                [
                    self.constructor_cls.__module__,
                    self.constructor_cls.__name__,
                    name,
                    self.conf.state,
                ]
            )
            names.append(hash.hexdigest())

        return names

    def download(self) -> None:
        """Download raw datasets."""

        for name in self.sets:
            download_url(self.DATASET_URLS[name], self.raw_dir)

    @delayed
    def process_sample(
        self,
        zf_path: Path,
        pr_subpath: Path,
        processed_dir: Path,
    ) -> None:
        """Process single protein-ligand sample.

        Arguments:
            zf_path: Path to zip-file containing all protein-ligand files.
            pr_subpath: Subpath of protein within zip-file that should be
                processed.
            processed_dir: Directory where to store processed samples.
        """

        with disable_warnings(disable=(self.verbose <= 1)):
            with TemporaryDirectory() as temp_dir:

                with ZipFile(zf_path) as zf:
                    for p in zf.namelist():
                        if p.startswith(str(pr_subpath)):
                            zf.extract(p, temp_dir)

                pr_path = temp_dir / pr_subpath
                constructor = self.constructor_cls(pr_path, self.conf)
                pt_path = processed_dir / f"{constructor.name}.pt"
                pk_path = processed_dir / f"{constructor.name}.meta.pkl"
                if not pt_path.exists():
                    data, meta = constructor.construct()
                    torch.save(data, pt_path)
                    with open(pk_path, "wb") as pf:
                        pickle.dump(meta, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def process(self) -> None:
        """Process raw samples."""

        if not self.raw_file_names:
            return

        process_args = set()

        for zf_name, ds_hash in zip(self.raw_file_names, self.processed_file_names):
            zf_path = Path(self.raw_dir) / zf_name
            processed_dir = Path(self.processed_dir) / ds_hash
            processed_dir.mkdir(exist_ok=True)

            with ZipFile(zf_path) as zf:
                for p in zf.namelist():
                    p = Path(p)
                    if p.name in ("protein.pdb", "protein.mol2"):
                        process_args.add((zf_path, p.parent, processed_dir))

        parallel = ProgressParallel(
            use_tqdm=(self.verbose == 1),
            total=len(process_args),
            n_jobs=self.n_jobs,
            verbose=False,
            timeout=None,
        )

        try:
            parallel(self.process_sample(*args) for args in process_args)
        except:
            if self.cleanup_on_error:
                shutil.rmtree(processed_dir)
            raise

    def len(self) -> int:
        """Number of samples in dataset."""

        return len(self._processed)

    def get(self, idx: int) -> Data:
        """Get sample from dataset.

        Arguments:
            idx: Index of sample.

        Returns:
            Sample with index `idx`.
        """

        return torch.load(self._processed[idx])

    def info(self, idx: int) -> dict[str, Any]:
        """Get additional sample information.

        Arguments:
            idx: Index of sample.

        Returns:
            Additional sample information.
        """

        path = self._processed[idx]
        with open(path.parent / f"{path.stem}.meta.pkl", "rb") as pf:
            return pickle.load(pf)


class PUResNetDataModule(LightningDataModule):
    """Lightning datamodule for the PUResNet dataset."""

    def __init__(
        self,
        train_sets: list[str],
        val_sets: list[str],
        test_sets: list[str],
        dataset_setup: Callable[[list[str]], Dataset],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        """
        Arguments:
            train_sets: List of datasets for training.
            val_sets: List of datasets for validation.
            test_sets: List of datasets for testing.
            dataset_setup: Callable that takes a list of requested datasets
                and returns a dataset instance with the respective datasets.
            batch_size: Batch size for dataloaders.
            shuffle: Whether batches in training dataloader should be shuffled.
            num_workers: Number of workers.
        """

        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "train_sets",
                "val_sets",
                "test_sets",
                "dataset_setup",
                "num_workers",
            ]
        )

        self.train_sets = train_sets
        self.val_sets = val_sets
        self.test_sets = test_sets
        self.dataset_setup = dataset_setup
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Download and process datasets."""

        sets = list({*self.train_sets, *self.val_sets, *self.test_sets})
        self.dataset_setup(sets)

    def setup(self, stage: str) -> None:
        """Setup datasets for training and testing.

        Arguments:
            stage: At which stage this method is called. With `"fit"` instances
                for the training and validation datasets will be created, with
                `"test"` an instance for the testing dataset.
        """

        match stage:
            case "fit":
                self.train_dataset = self.dataset_setup(self.train_sets)
                self.val_dataset = self.dataset_setup(self.val_sets)
            case "test":
                self.test_dataset = self.dataset_setup(self.test_sets)

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader.

        Returns:
            Training dataloader.
        """

        return DataLoader(
            self.train_dataset,
            self.batch_size,
            self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader.

        Returns:
            Validation dataloader.
        """

        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the testing dataloader.

        Returns:
            Testing dataloader.
        """

        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
