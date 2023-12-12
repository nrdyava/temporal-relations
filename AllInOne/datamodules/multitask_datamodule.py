import functools
import numpy as np
from . import _datamodules
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()
        self.dm = _datamodules['activitynet'](_config)
        self.batch_size = self.dm.batch_size
        self.vocab_size = self.dm.vocab_size
        self.num_workers = self.dm.num_workers
        self.dist = dist

    def prepare_data(self):
        self.dm.prepare_data()

    def setup(self, stage):
        self.dm.setup(stage)
        self.train_dataset = self.dm.train_dataset
        self.val_dataset = self.dm.val_dataset
        self.test_dataset = self.dm.test_dataset
        self.tokenizer = self.dm.tokenizer

        self.collate = functools.partial(
            self.dm.train_dataset.collate, 
            mlm_collator=self.dm.mlm_collator,
        )

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            collate_fn=self.collate,
            persistent_workers=True,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            collate_fn=self.collate,
            persistent_workers=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            collate_fn=self.collate,
            persistent_workers=True,
        )
        return loader
