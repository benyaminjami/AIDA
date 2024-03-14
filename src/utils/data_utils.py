import heapq
from typing import Any, Callable, Iterator, List, TypeVar

import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch import distributed as dist
from torch.utils.data import DataChunk
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from . import esm


class Alphabet:
    def __init__(self, name="esm", featurizer="mpnn", alphabet_cfg={}, featurizer_cfg={}):
        self.name = name
        self._alphabet = None

        if name == "esm":
            self._alphabet = self._alphabet = esm.Alphabet.from_architecture("ESM-1b")
            self.add_special_tokens = True
        elif name == "gearnet":
            self._alphabet = self._alphabet = esm.Alphabet.from_architecture("ESM-1b")
            self.add_special_tokens = True
            self.prepend_bos = False
            self.append_eos = False
        elif name == "paired_balm":
            self._alphabet = self._alphabet = esm.Alphabet.from_architecture("paired_balm")
            self.add_special_tokens = True
        elif name == "mpnn":
            self._alphabet = esm.Alphabet(
                standard_toks=[
                    "A",
                    "R",
                    "N",
                    "D",
                    "C",
                    "Q",
                    "E",
                    "G",
                    "H",
                    "I",
                    "L",
                    "K",
                    "M",
                    "F",
                    "P",
                    "S",
                    "T",
                    "W",
                    "Y",
                    "V",
                ],
                prepend_toks=["<pad>", "<unk>"],
                append_toks=[],
                prepend_bos=False,
                append_eos=False,
            )
        else:
            raise NotImplementedError(name)

        self._featurizer = self.get_featurizer(featurizer, **featurizer_cfg)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._alphabet, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__} has no attribute `{name}`.")

    def __len__(self):
        return len(self._alphabet)

    def get_featurizer(self, name="cath", **kwds):
        if name == "balm":
            from src.data.datasets.balm_batcher import AntibodyFeaturizer as Featurizer
            return Featurizer(alphabet=self)

        elif name == "mpnn":
            from src.data.datasets.mpnn_batcher import AntigenFeaturizer as Featurizer
            return Featurizer(alphabet=self, coord_nan_to_zero=kwds.get("coord_nan_to_zero", True))

        elif name == 'gearnet':
            from src.data.datasets.gearnet_batcher import AntigenFeaturizer as Featurizer
            return Featurizer(alphabet=self)

    @property
    def featurizer(self):
        return self._featurizer

    def featurize(self, raw_batch, **kwds):
        return self._featurizer(raw_batch, **kwds)

    def decode(self, tensors, return_as="str", remove_special=False):
        ret = []
        for i, ids in enumerate(tensors.cpu()):
            if return_as == "str":
                line = "".join([self.get_tok(id) for id in ids])
                if remove_special:
                    line = (
                        line.replace(self.get_tok(self.mask_idx), "_")
                        .replace(self.get_tok(self.eos_idx), "")
                        .replace(self.get_tok(self.cls_idx), "")
                        .replace(self.get_tok(self.padding_idx), "")
                        .replace(self.get_tok(self.unk_idx), "-")
                    )
            elif return_as == "list":
                line = [self.get_tok(id) for id in ids]
            ret.append(line)
        return ret


T_co = TypeVar("T_co", covariant=True)


def identity(example):
    return example


class MaxTokensBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        max_tokens=6000,
        drop_last=False,
        distributed=False,
        sort_key: Callable = None,
        sort=False,
        buffer_size_multiplier=100,
        seed=42,
        shuffle=True,
        sampler=None,
    ):
        self.distributed = distributed
        if sampler is None:
            if distributed:
                sampler = DistributedSampler(dataset, shuffle=shuffle)
            else:
                sampler = RandomSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)

        self.max_tokens = max_tokens
        self.sort_key = sort_key
        self.sort = sort
        self.max_buffer_size = max_tokens * buffer_size_multiplier

        self._epoch = 0
        self._seed = seed
        self.shuffle = shuffle

        self.bucket_batches = []
        self._num_samples = None
        self._build_batches()

    def __len__(self):
        return len(self.bucket_batches)

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        for batch, batch_size in self.bucket_batches:
            yield batch

    def _build_batches(self):
        buffer: List = []
        buffer_size: int = 0

        batch: List = []
        batch_size: int = 0

        bucket_batches = []

        if self.sort:
            indices = sorted(list(self.sampler), key=self.sort_key)
        else:
            indices = self.sampler

        for index in indices:
            # 1) add to buffer
            length = self.sort_key(index)
            # heapq.heappush(buffer, (length, index))
            buffer.append((length, index))
            buffer_size += length

            # 2) create batches in the sorted buffer once buffer is full
            if buffer_size > self.max_buffer_size:
                # length, index = heapq.heappop(buffer)
                length, index = buffer.pop()
                buffer_size -= length
                if batch_size + length > self.max_tokens:
                    bucket_batches.append((batch, batch_size))
                    batch, batch_size = [], 0
                batch.append(index)
                batch_size += length

        # 3) create batches from the rest data in the buffer
        while buffer:
            # length, index = heapq.heappop(buffer)
            length, index = buffer.pop()
            # print(length, index)
            if batch_size + length > self.max_tokens:
                bucket_batches.append((batch, batch_size))
                batch, batch_size = [], 0
            batch.append(index)
            batch_size += length

        # 4) the last batch
        if batch:
            bucket_batches.append((batch, batch_size))

        # 5) maybe shuffle
        if self.shuffle:
            seed_everything(self._seed + self._epoch)
            np.random.shuffle(bucket_batches)

        # 6) carefully deal with DDP, ensuring that every worker
        #      has the same number of batches
        if self.distributed:
            num_samples = torch.tensor(len(bucket_batches)).to(self.sampler.rank)
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            num_samples = num_samples.item()

            if len(bucket_batches) < num_samples:
                padding_size = num_samples - len(bucket_batches)
                bucket_batches += bucket_batches[:padding_size]

        self.bucket_batches = bucket_batches

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distributed:
            self.sampler.set_epoch(epoch)
