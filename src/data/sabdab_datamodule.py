from functools import partial
from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import RankedLogger

from ..utils.data_utils import Alphabet, MaxTokensBatchSampler
from .datasets.sabdab_dataset import SAbDab, collate_fn

log = RankedLogger(__name__, rank_zero_only=True)


class SAbDabDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        max_length: int = 500,
        atoms: List[str] = ("N", "CA", "C", "O"),
        alphabet=None,
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = "train",
        valid_split: str = "valid",
        test_split: str = "test",
        debug: bool = False,
        verbose: bool = False,
        truncate: int = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == "fit":
            if self.hparams.debug:
                self.hparams.train_split = "valid"
            self.train_dataset, _ = SAbDab(
                self.hparams.data_dir,
                max_length=self.hparams.max_length,
                split=self.hparams.train_split,
                truncate=self.hparams.truncate,
                verbose=self.hparams.verbose,
            )

            self.valid_dataset, _ = SAbDab(
                self.hparams.data_dir,
                max_length=self.hparams.max_length,
                split=self.hparams.valid_split,
                truncate=self.hparams.truncate,
                verbose=self.hparams.verbose,
            )

        self.test_dataset, _ = SAbDab(
            self.hparams.data_dir,
            max_length=self.hparams.max_length,
            split=self.hparams.test_split,
            truncate=self.hparams.truncate,
            verbose=self.hparams.verbose,
        )

        self.alphabet_antigen = Alphabet(**self.hparams.alphabet.encoder)
        self.alphabet_antibody = Alphabet(**self.hparams.alphabet.decoder)

        self.collate_batch = partial(
            collate_fn,
            antigen_featurizer=self.alphabet_antigen.featurizer,
            antibody_featurizer=self.alphabet_antibody.featurizer,
        )

    def _build_batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        # build batch sampler
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size,
            max_tokens=max_tokens,
            sort=self.hparams.sort,
            drop_last=False,
            sort_key=lambda i: len(dataset[i][0]["seqs"][0]) + len(dataset[i][0]["seqs"][1]),
        )
        return batch_sampler

    def train_dataloader(self):
        if not hasattr(self, "train_batch_sampler"):
            self.train_batch_sampler = self._build_batch_sampler(
                self.train_dataset, max_tokens=self.hparams.max_tokens, shuffle=True
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset=self.valid_dataset,
                batch_sampler=self._build_batch_sampler(
                    self.valid_dataset,
                    max_tokens=self.hparams.max_tokens,
                    distributed=False,
                ),
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_batch,
            ),
            self.test_dataloader(),
        ]

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=self._build_batch_sampler(
                self.test_dataset, max_tokens=self.hparams.max_tokens, distributed=False
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch,
        )
