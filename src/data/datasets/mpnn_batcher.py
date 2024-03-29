from typing import Sequence, Tuple

import numpy as np
import torch

from src.utils.data_utils import Alphabet


class BatchConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a processed (labels + tensor)
    batch."""

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list
            ]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos): len(seq_encoded) + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_encoded) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens


class AntigenBatchConverter(BatchConverter):
    def __init__(
        self,
        alphabet,
        coord_nan_to_zero=True,
    ):
        super().__init__(alphabet)
        self.coord_nan_to_zero = coord_nan_to_zero

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x n_atoms x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        # self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, dists, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.0
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = "X" * len(coords)
            batch.append(((coords, dists, confidence), seq))

        coords_dists_confidence, strs, tokens = super().__call__(batch)

        coords = [torch.tensor(cd) for cd, _, _ in coords_dists_confidence]
        dists = [torch.tensor(dt) for _, dt, _ in coords_dists_confidence]
        confidence = [torch.tensor(cf) for _, _, cf in coords_dists_confidence]

        coords = collate_dense_tensors(coords, pad_v=np.nan)
        dists = collate_dense_tensors(dists, pad_v=1000)
        confidence = collate_dense_tensors(confidence, pad_v=-1.0)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords = coords.to(device)
            dists = dists.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])

        coord_mask = torch.isfinite(coords.sum([-2, -1]))  # & tokens.ne(self.alphabet.unk_idx)
        confidence = dists * coord_mask + (1000.0) * coord_padding_mask
        confidence = confidence * coord_mask + (-1.0) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.0

        return coords, dists, confidence, strs, tokens, lengths, coord_mask

    def from_lists(
        self, coords_list, dists=None, confidence_list=None, seq_list=None, device=None
    ):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            dists: list of distances of each aminoacid to paratope
            floats in shape L
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            raise
        raw_batch = zip(coords_list, dists, confidence_list, seq_list)
        return self.__call__(raw_batch, device)


class AntigenFeaturizer:
    def __init__(
        self,
        alphabet: Alphabet,
        coord_nan_to_zero=True,
        atoms=("N", "CA", "C", "O"),
    ):
        self.alphabet = alphabet
        self.batcher = AntigenBatchConverter(
            alphabet=alphabet,
            coord_nan_to_zero=coord_nan_to_zero,
        )

        self.atoms = atoms

    def __call__(self, raw_batch: dict):
        seqs, coords, names, dists = [], [], [], []
        for entry in raw_batch:
            # [L, 3] x 4 -> [L, 4, 3]
            if isinstance(entry["coords"], dict):
                coords.append(np.stack([entry["coords"][atom] for atom in self.atoms], 1))
            else:
                coords.append(entry["coords"])
            seqs.append(entry["seqs"].replace('X', '<unk>'))
            names.append(entry["pdb"])
            dists.append(entry["dists"])

        (
            coords,
            dists,
            confidence,
            strs,
            tokens,
            lengths,
            coord_mask,
        ) = self.batcher.from_lists(
            coords_list=coords, dists=dists, confidence_list=None, seq_list=seqs
        )

        assert coords.size(1) == tokens.size(1) == dists.size(1), names

        # coord_mask = coord_mask > 0.5
        batch = {
            "coords": coords,
            "dists": dists,
            "tokens": tokens,
            "confidence": confidence,
            "coord_mask": coord_mask,
            "lengths": lengths,
            "seqs": seqs,
            "names": names,
        }
        return batch


def collate_dense_tensors(samples, pad_v):
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
                      ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len({x.dim() for x in samples}) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    (device,) = tuple({x.device for x in samples})  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def new_arange(x, *size):
    """Return a Tensor of `size` filled with a range function on the device of x.

    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()
