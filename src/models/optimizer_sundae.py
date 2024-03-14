import torch

from src.utils.data_utils import Alphabet
import random
from math import comb


def new_arange(x, *size):
    """Return a Tensor of `size` filled with a range function on the device of x.

    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores)) * p
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def generate_combinations(batch_antibody, num_combinations=100, t=None, mask_id=32):
    noise_mask = batch_antibody["noise_mask"]
    true_indices = torch.nonzero(noise_mask).squeeze()
    num_true = true_indices.size(0)

    # Check if the number of True values is less than t
    if num_true < t:
        t = num_true

    # Check if the number of combinations is less than num_combinations
    if t != 0 and comb(num_true, t) < num_combinations:
        num_combinations = comb(num_true, t)

    combinations = set()
    while len(combinations) < num_combinations:
        n_mask = t if t != 0 else random.randint(0, num_true)
        selected_indices = true_indices[torch.randperm(num_true)[:n_mask]]
        new_tensor = torch.zeros_like(noise_mask).squeeze()
        new_tensor[selected_indices[:, 1]] = 1
        # Convert tensor to a binary number and add it to the set of combinations
        combinations.add(new_tensor)

    # Convert back to tensor format
    prev_token_mask = torch.stack(list(combinations), 0)
    prev_tokens = batch_antibody["tokens"].squeeze().repeat(num_combinations, 1)
    prev_tokens = prev_tokens.masked_fill(prev_token_mask, mask_id)
    # batch_antibody["prev_token_mask"] = torch.stack(list(combinations), 0)
    # batch_antibody["prev_tokens"] = (
    #     batch_antibody["tokens"].squeeze().repeat(num_combinations, 1)
    # )
    # batch_antibody["prev_tokens"] = batch_antibody["prev_tokens"].masked_fill(
    #     batch_antibody["prev_token_mask"], mask_id
    # )
    return prev_tokens, prev_token_mask


class AntibodyOptimizer:
    def __init__(
        self,
        alphabet: Alphabet = None,
        n_samples: int = 100,
        temperature: float = None,
        batch_size: int = 15,
        t: int = 0,
        **kwargs
    ):
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.temperature = temperature
        self.n_samples = n_samples
        self.t = t
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(
        self,
        model,
        batch: dict,
        alphabet: Alphabet = None,
        max_iter: int = None,
        strategy: str = None,
        temperature: float = None,
        need_attn_weights: bool = False,
        replace_visible_tokens: bool = True,
    ):
        alphabet = alphabet or self.alphabet
        max_iter = max_iter or self.max_iter
        strategy = strategy
        temperature = temperature or self.temperature

        batch_antibody = batch["antibody"]
        batch_antigen = batch["antigen"]
        assert batch_antibody["tokens"].size(0) == 1, "More than one sample"

        results = []

        # 0) encoding
        encoder_out = model.forward_encoder(batch_antigen)
        tokens = batch_antibody["tokens"][0].repeat(self.n_samples, 1)
        masks = batch_antibody["prev_token_mask"][0].repeat(self.n_samples, 1)
        initial_output_tokens_mask = batch_antibody["prev_token_mask"]

        # iterative refinement
        for batch in range(1 + (self.n_samples // self.batch_size)):
            start, end = batch * self.batch_size, min(
                self.n_samples, (batch + 1) * self.batch_size
            )
            batch_size = end - start
            batch_antibody["prev_tokens"] = tokens[start:end]
            batch_antibody["prev_token_mask"] = masks[start:end]
            batch_antibody["chain_ids"] = batch_antibody["chain_ids"][0].repeat(
                batch_size, 1
            )
            batch_antibody["attention_mask"] = batch_antibody["attention_mask"][
                0
            ].repeat(batch_size, 1)
            batch_antibody["position_ids"] = batch_antibody["position_ids"][0].repeat(
                batch_size, 1
            )
            encoder_out["feats"] = encoder_out["feats"][0].repeat(batch_size, 1, 1)
            encoder_out["feats_mask"] = encoder_out["feats_mask"][0].repeat(
                batch_size, 1
            )
            initial_output_tokens_mask = initial_output_tokens_mask[0].repeat(
                batch_size, 1
            )
            for step in range(10):
                # 2.1: predict
                decoder_out = model.forward_decoder(
                    batch_antibody=batch_antibody,
                    batch_antigen=encoder_out,
                    need_attn_weights=need_attn_weights,
                )
                output_tokens, output_scores = sample_from_categorical(
                    decoder_out, temperature=self.temperature
                )

                # 2.3: update
                if replace_visible_tokens:
                    visible_token_mask = ~batch_antibody["prev_token_mask"]
                    visible_tokens = batch_antibody["prev_tokens"]
                    output_tokens = torch.where(
                        visible_token_mask, visible_tokens, output_tokens
                    )
                    output_scores = torch.where(
                        visible_token_mask,
                        torch.zeros_like(batch_antibody["tokens"]),
                        output_scores,
                    )

                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=initial_output_tokens_mask,
                    p=1 - (step + 1) / 10,
                )
                output_tokens[
                    ~(~skeptical_mask * initial_output_tokens_mask)
                ] = batch_antibody["prev_tokens"][
                    ~(~skeptical_mask * initial_output_tokens_mask)
                ]
                results.append(output_tokens)
                if step + 1 == max_iter:
                    break
        results = torch.cat(results, 0)

        return results, None, None


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    elif temperature is None or temperature == 0.0:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
