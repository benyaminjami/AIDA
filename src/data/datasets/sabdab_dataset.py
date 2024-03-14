import json
import os
from typing import Dict, List

import numpy as np

from src.utils import RankedLogger

# import esm

log = RankedLogger(__name__, rank_zero_only=True)


def SAbDab(
    root=".data/Rabd",
    split="train",
    truncate=None,
    max_length=500,
    alphabet="ACDEFGHIKLMNPQRSTVWYX",
    verbose=False,
):
    alphabet_set = {a for a in alphabet}
    ab_path = os.path.join(root, f"{split}.ab-ag.ab.json")
    ag_path = os.path.join(root, f"{split}.ab-ag.ag.json")

    discard_count = {"bad_chars": 0, "too_long": 0, "no_coords": 0, "repeat": 0}
    repeat_checker = []
    # 1) load the dataset
    with open(ab_path) as f_ab, open(ag_path) as f_ag:
        dataset: List[Dict] = []

        ab_lines = f_ab.readlines()
        ag_lines = f_ag.readlines()
        for i, (ab_line, ag_line) in enumerate(zip(ab_lines, ag_lines)):
            # if i > 300: break
            ab_entry = json.loads(ab_line)
            ag_entry = json.loads(ag_line)
            name = ab_entry["pdb"]

            if ag_entry["coords"] == []:
                if verbose:
                    print(name, "No coords")
                discard_count["no_coords"] += 1
                continue

            select_antigen_chain(ag_entry)
            check(ab_entry=ab_entry, ag_entry=ag_entry)
            if len(ag_entry["seqs"]) > max_length:
                select_antigen_contact_range(ag_entry, max_length)

            ab_seqs = ab_entry["seqs"]
            ag_seqs = ag_entry["seqs"]

            # Convert raw coords to np arrays
            for key, val in ag_entry["coords"].items():
                ag_entry["coords"][key] = np.asarray(val, dtype=np.float32)

            # Check if in alphabet
            bad_chars = set(
                [s for seq in ab_seqs for s in seq]
                + [s for seq in ag_seqs for s in seq]
            ).difference(alphabet_set)

            if len(bad_chars) == 0:
                if (
                    ab_entry["pdb"],
                    ab_entry["seqs"][1],
                    ab_entry["seqs"][0],
                ) in repeat_checker:
                    discard_count["repeat"] += 1
                elif (
                    len(ag_entry["seqs"]) <= max_length
                    and len(ab_entry["seqs"][0]) + len(ab_entry["seqs"][1]) < 2 * 168
                ):  # TODO clean this part
                    dataset.append((ab_entry, ag_entry))
                    repeat_checker.append(
                        (ab_entry["pdb"], ab_entry["seqs"][1], ab_entry["seqs"][0])
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

        return dataset, alphabet_set


def select_antigen_chain(entry):
    contacted_chain_idx = np.array(
        list(map(sum, map(lambda x: x < 8, (map(np.array, entry["dists"])))))
    ).argmax()
    del entry["pdb_data_path"]
    for key, values in entry.items():
        if key == "pdb":
            continue
        entry[key] = values[contacted_chain_idx]


def check(ab_entry, ag_entry):
    assert (
        len(ab_entry["weights"][0])
        == len(ab_entry["seqs"][0])
        == len(ab_entry["position_ids"][0])
        == len(ab_entry["dists"][0])
    ), "Mismatch length in Light chain"
    assert (
        len(ab_entry["weights"][1])
        == len(ab_entry["seqs"][1])
        == len(ab_entry["position_ids"][1])
        == len(ab_entry["dists"][1])
    ), "Mismatch length in Heavy chain"
    assert (
        len(ag_entry["seqs"])
        == len(ag_entry["dists"])
        == len(ag_entry["coords"]["N"])
        == len(ag_entry["coords"]["CA"])
        == len(ag_entry["coords"]["C"])
        == len(ag_entry["coords"]["O"])
    ), "Mismatch length in Antigen chain"


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
    for key, values in entry["coords"].items():
        entry["coords"][key] = values[start:end]
    for key in ["dists", "seqs"]:
        entry[key] = entry[key][start:end]


def collate_fn(batch, antigen_featurizer, antibody_featurizer):
    antibodies = [b[0] for b in batch]
    antigens = [b[1] for b in batch]
    batched_antibody = antibody_featurizer(antibodies)
    batched_antigen = antigen_featurizer(antigens)
    batch = {"antibody": batched_antibody, "antigen": batched_antigen}
    return batch
