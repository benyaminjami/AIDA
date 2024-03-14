from typing import Sequence, Tuple
import random
import torch
from torchdrug import layers, data
from torchdrug.layers import geometry

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
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
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
                int(self.alphabet.prepend_bos): len(seq_encoded)
                + int(self.alphabet.prepend_bos),
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
    ):
        super().__init__(alphabet)

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
        for dists, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.0
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(dists)
            if seq is None:
                seq = "X" * len(dists)
            batch.append(((dists, confidence), seq))

        dists_confidence, strs, tokens = super().__call__(batch)

        dists = [torch.tensor(dt) for dt, _ in dists_confidence]
        confidence = [torch.tensor(cf) for _, cf in dists_confidence]

        dists = collate_dense_tensors(dists, pad_v=1000)
        confidence = collate_dense_tensors(confidence, pad_v=-1.0)

        mask = tokens.ne(self.alphabet.padding_idx)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            dists = dists.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            mask = mask.to(mask)

        return dists, confidence, strs, tokens, lengths, mask

    def from_lists(self, dists=None, confidence_list=None, seq_list=None, device=None):
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
        batch_size = len(dists)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            raise
        raw_batch = zip(dists, confidence_list, seq_list)
        return self.__call__(raw_batch, device)


class AntigenFeaturizer:
    def __init__(
        self,
        alphabet: Alphabet,
        atoms=("N", "CA", "C", "O"),
    ):
        self.alphabet = alphabet
        self.batcher = AntigenBatchConverter(
            alphabet=alphabet,
        )

        self.atoms = atoms

        self.graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
                geometry.SequentialEdge(max_distance=2),
            ],
            edge_feature="gearnet",
        )

    def __call__(self, raw_batch: dict):
        seqs, graphs, names, dists = [], [], [], []
        for entry in raw_batch:
            start, end = None, None
            entry["seqs"] = entry["seqs"].replace('X', '<unk>')
            graph = entry["graph"].clone()
            if len(entry["seqs"]) > 350:
                start = random.randint(0, len(entry["seqs"]) - 350)
                end = start + 350
                mask = torch.zeros(
                    entry["graph"].num_residue,
                    dtype=torch.bool,
                    device=entry["graph"].device,
                )
                mask[start:end] = True
                graph = graph.subresidue(mask)  # Graph Truncate Transformation

            graph.view = "residue"  # Graph View Transformation
            graphs.append(graph)
            seqs.append(entry["seqs"][start:end])
            names.append(entry["pdb"])
            dists.append(entry["dists"][start:end])

        (
            dists,
            confidence,
            strs,
            tokens,
            lengths,
            mask,
        ) = self.batcher.from_lists(dists=dists, confidence_list=None, seq_list=seqs)
        assert tokens.size(1) == dists.size(1), names
        assert all(
            mask[i].sum() == graph.num_residue for i, graph in enumerate(graphs)
        ), names

        graphs = data.Protein.pack(graphs)
        graphs = self.graph_construction_model(graphs)

        batch = {
            "graphs": graphs,
            "dists": dists,
            "tokens": tokens,
            "confidence": confidence,
            "mask": mask,
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
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple({x.device for x in samples})  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
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
