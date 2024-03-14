import json
from tqdm import tqdm
import pickle
import os
from typing import Dict, List
from torchdrug.data.protein import Protein
import numpy as np
import torch
from src.utils import RankedLogger
from torchdrug.layers import geometry

# import esm

log = RankedLogger(__name__, rank_zero_only=True)


def SAbDab(
    root=".data/Rabd",
    split="train",
    truncate=None,
    max_length=350,
    alphabet="ACDEFGHIKLMNPQRSTVWYX",
    verbose=False,
    mode="hl",
    filter="h*",
):
    alphabet_set = {a for a in alphabet}
    ab_path = os.path.join(root, f"{split}.ab-ag.ab.json")
    ag_path = os.path.join(root, f"{split}.ab-ag.ag.json")
    ca_filter_transform = geometry.AlphaCarbonNode()
    discard_count = {"bad_chars": 0, "too_long": 0, "no_coords": 0, "repeat": 0}
    repeat_checker = []
    dataset: List[Dict] = []
    if os.path.exists(os.path.join(root, f"{split}.pkl")):
        with open(os.path.join(root, f"{split}.pkl"), "rb") as f:
            pkl_dataset = pickle.load(f)
        for i, (ab_entry, ag_entry) in enumerate(pkl_dataset):
            if resolve_mode(ab_entry, mode) is None:
                continue
            # if i > 34 and i < 70:
            # if ab_entry['pdb'] == '4g6m' or ab_entry['pdb'] == '2cmr':
            dataset.append((ab_entry, ag_entry))
        log.info(f"Loaded data size: {len(dataset)}/{len(pkl_dataset)}")
        if truncate:
            dataset = dataset[:truncate]
        return dataset, alphabet_set
    # 1) load the dataset
    with open(ab_path) as f_ab, open(ag_path) as f_ag:
        ab_lines = f_ab.readlines()
        ag_lines = f_ag.readlines()
        for i, (ab_line, ag_line) in enumerate(tqdm(zip(ab_lines, ag_lines))):
            ab_entry = json.loads(ab_line)
            ag_entry = json.loads(ag_line)
            if resolve_mode(ab_entry, mode) is None:
                continue
            name = ab_entry["pdb"]

            ag_entry.pop("encoded_seq", None)
            ab_entry.pop("encoded_seq", None)
            try:
                antigen_graph = Protein.from_pdb(
                    ag_entry["pdb_data_path"], atom_feature=None, bond_feature=None
                )
            except ValueError:
                discard_count["no_coords"] += 1
                continue
            ag_entry["graph"] = ca_filter_transform(antigen_graph)

            select_antigen_chain(ag_entry)
            remove_nan(ag_entry)
            check_lens(ab_entry, ag_entry)
            if len(ag_entry["seqs"]) > max_length:
                select_antigen_contact_range(ag_entry, max_length)

            # ag_entry = transform(ag_entry)

            ab_seqs = ab_entry["seqs"]
            ag_seqs = ag_entry["seqs"]
            ag_entry.pop("coords", None)

            # Check if in alphabet
            bad_chars = set(
                [s for seq in ab_seqs for s in seq]
                + [s for seq in ag_seqs for s in seq]
            ).difference(alphabet_set)

            if len(bad_chars) == 0:
                if (
                    ab_entry["pdb"],
                    "".join(ab_entry["seqs"])
                ) in repeat_checker:
                    discard_count["repeat"] += 1
                elif (
                    len(ag_entry["seqs"]) <= max_length
                ):  # TODO clean this part
                    dataset.append((ab_entry, ag_entry))
                    repeat_checker.append(
                        (ab_entry["pdb"], "".join(ab_entry["seqs"]))
                    )
                else:
                    discard_count["too_long"] += 1
            else:
                if verbose:
                    print(name, bad_chars)
                    print(ag_entry["seqs"])
                    print(ab_entry["seqs"])
                discard_count["bad_chars"] += 1

            if verbose and (i + 1) % 100000 == 0:
                print(f"{len(dataset)} entries ({i + 1} loaded)")

            # Truncate early
            if truncate is not None and len(dataset) == truncate:
                break
        total_size = i

        log.info(
            f"Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}."
        )
        with open(os.path.join(root, f"{split}.pkl"), "wb") as f:
            pickle.dump(dataset, f)
        return dataset, alphabet_set


def resolve_mode(ab_entry, mode):
    if mode == 'h*':
        return ab_entry
    if mode == 'h':
        if len(ab_entry['seqs']) == 1:
            return ab_entry
        for key in ["weights", "seqs", "position_ids", "dists"]:
            ab_entry[key] = [ab_entry[key][1]]
        return ab_entry
    if mode == 'hl':
        if len(ab_entry['seqs']) == 1:
            return None
        return ab_entry


def check_lens(ab_entry, ag_entry):
    assert (
        len(ab_entry["weights"][0])
        == len(ab_entry["seqs"][0])
        == len(ab_entry["position_ids"][0])
        == len(ab_entry["dists"][0])
    ), "Mismatch length in Light chain" + str(ab_entry)
    if len(ab_entry['seqs']) == 2:
        assert (
            len(ab_entry["weights"][1])
            == len(ab_entry["seqs"][1])
            == len(ab_entry["position_ids"][1])
            == len(ab_entry["dists"][1])
        ), "Mismatch length in Heavy chain" + str(ab_entry)
    assert (
        len(ag_entry["seqs"]) == len(ag_entry["dists"]) == ag_entry["graph"].num_residue
    ), "Mismatch length in Antigen chain"


def select_antigen_chain(entry):
    contacted_chain_idx = np.array(
        list(map(sum, map(lambda x: x < 8, (map(np.array, entry["dists"])))))
    ).argmax()
    del entry["pdb_data_path"]
    for key, values in entry.items():
        if key in ["pdb", "graph"]:
            continue
        entry[key] = values[contacted_chain_idx]

    chain = entry["chains"]
    mapping = {}
    # Map uppercase letters A-Z to numbers 1-26
    for i, letter in enumerate(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 1
    ):
        mapping[letter] = i
    chain_id = mapping[chain]
    entry["graph"] = entry["graph"].subresidue(entry["graph"].chain_id == chain_id)


def remove_nan(entry):
    mask = ~np.isnan(np.array(entry['coords']['CA']).sum(-1))
    entry["seqs"] = "".join(list(np.array(list(entry["seqs"]))[mask]))
    entry["dists"] = list(np.array(entry["dists"])[mask])


def select_antigen_contact_range(entry, max_len):
    dists = np.array(entry["dists"])
    contacts = dists < 8
    if len(dists) < max_len:
        raise ValueError("The dists length is smaller than the max_len.")

    # Initialize the current sum and the minimum sum
    current_contact = np.sum(contacts[:max_len])
    max_contact = current_contact
    currect_sum = np.sum(dists[:max_len])
    min_sum = currect_sum

    min_start_index = 0

    # Slide the window
    for i in range(1, len(dists) - max_len + 1):
        current_contact = current_contact - contacts[i - 1] + contacts[i + max_len - 1]
        currect_sum = currect_sum - dists[i - 1] + dists[i + max_len - 1]
        if current_contact >= max_contact and currect_sum < min_sum:
            max_contact = current_contact
            min_start_index = i

    start, end = min_start_index, min_start_index + max_len
    for key in ["dists", "seqs"]:
        entry[key] = entry[key][start:end]

    graph = entry["graph"]
    mask = torch.zeros(graph.num_residue, dtype=torch.bool, device=graph.device)
    mask[start:end] = True
    entry["graph"] = graph.subresidue(mask)


def collate_fn(batch, antigen_featurizer, antibody_featurizer):
    antibodies = [b[0] for b in batch]
    antigens = [b[1] for b in batch]
    batched_antibody = antibody_featurizer(antibodies)
    batched_antigen = antigen_featurizer(antigens)
    batch = {"antibody": batched_antibody, "antigen": batched_antigen}
    return batch
