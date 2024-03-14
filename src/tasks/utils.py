import torch


@torch.no_grad()
def inject_noise(
    tokens,
    attention_mask,
    alphabet,
    noise,
    sel_mask=None,
    sel_mask_add_prob=0.1,
    keep_idx=None,
    keep_special_tokens=True,
):
    tokens = tokens.clone()
    padding_idx = alphabet.padding_idx
    if keep_idx is None:
        keep_idx = [alphabet.cls_idx, alphabet.eos_idx]

    def get_random_text(shape):
        return torch.randint(
            alphabet.get_idx(alphabet.standard_toks[0]),
            alphabet.get_idx(alphabet.standard_toks[-1]) + 1,
            shape,
        )

    mask_idx = alphabet.mask_idx

    def _full_mask(target_tokens):
        target_mask = (
            target_tokens.ne(padding_idx)  # & mask
            & target_tokens.ne(alphabet.cls_idx)
            & target_tokens.ne(alphabet.eos_idx)
            & attention_mask
        )
        # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
        masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
        return masked_target_tokens, target_mask

    def _random_mask(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        # target_score = target_score - (sel_mask * sel_mask_add_prob * 0.5)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.
        target_length = target_length - (2 * (target_length >= target_masks.sum(1)))
        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        masked_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
        )
        return (
            masked_target_tokens,
            target_cutoff.scatter(1, target_rank, target_cutoff) & attention_mask,
        )

    def _selected_mask(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask & sel_mask.ne(0)
        masked_target_tokens = torch.masked_fill(
            target_tokens, mask=target_masks, value=mask_idx
        )
        return masked_target_tokens, target_masks

    def _full_random(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask
        random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
        return (
            target_masks * random_text + ~target_masks * target_tokens,
            target_masks,
        )

    def _selected_random(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask & sel_mask.ne(0)
        random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
        return (
            target_masks * random_text + ~target_masks * target_tokens,
            target_masks,
        )

    def _sundae(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask
        corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1))
        rand = torch.rand(target_tokens.shape)
        mask = (rand < corruption_prob_per_sequence).to(
            target_tokens.device
        ) & target_masks

        random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
        return mask * random_text + ~mask * target_tokens, mask

    def _selected_sundae(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask & sel_mask.ne(0)
        corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1))
        rand = torch.rand(target_tokens.shape)
        mask = (rand < corruption_prob_per_sequence).to(
            target_tokens.device
        ) & target_masks

        random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
        return mask * random_text + ~mask * target_tokens, mask

    def _selected_guided_sundae(target_tokens):
        target_masks = target_tokens.ne(padding_idx) & attention_mask
        corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1)).to(
            target_tokens.device
        )
        rand = torch.rand(target_tokens.shape).to(target_tokens.device)
        rand = rand - sel_mask * sel_mask_add_prob
        mask = (rand < corruption_prob_per_sequence).to(
            target_tokens.device
        ) & target_masks

        random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
        return mask * random_text + ~mask * target_tokens, mask

    if noise == "no_noise":
        randomed_tokens, randomed_tokens_mask = tokens, sel_mask
    elif noise == "sundae":
        randomed_tokens, randomed_tokens_mask = _sundae(tokens)
    elif noise == "selected_sundae":
        randomed_tokens, randomed_tokens_mask = _selected_sundae(tokens)
    elif noise == "selected_guided_sundae":
        randomed_tokens, randomed_tokens_mask = _selected_guided_sundae(tokens)
    elif noise == "full_random":
        randomed_tokens, randomed_tokens_mask = _full_random(tokens)
    elif noise == "selected_random":
        randomed_tokens, randomed_tokens_mask = _selected_random(tokens)
    elif noise == "random_mask":
        randomed_tokens, randomed_tokens_mask = _random_mask(tokens)
    elif noise == "selected_mask":
        randomed_tokens, randomed_tokens_mask = _selected_mask(tokens)
    elif noise == "full_mask":
        randomed_tokens, randomed_tokens_mask = _full_mask(tokens)
    else:
        raise ValueError(f"Noise type ({noise}) not defined.")

    keep_mask = torch.isin(tokens, torch.tensor(keep_idx).to(tokens.device))
    if keep_special_tokens:
        prev_tokens = tokens * keep_mask + randomed_tokens * ~keep_mask
        prev_token_mask = randomed_tokens_mask & attention_mask & ~keep_mask
    else:
        prev_tokens = randomed_tokens
        prev_token_mask = randomed_tokens_mask & attention_mask

    return prev_tokens, prev_token_mask


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def resolve_noise_mask(batch_antibody, noise_mask):
    if batch_antibody.get(noise_mask, None) is not None:
        return batch_antibody[noise_mask]
    chain, cdr = noise_mask[0], int(noise_mask[1])
    assert chain in ['h', 'l']
    chain = 1 if chain == 'h' else 2
    return (batch_antibody['chain_ids'] == chain) & (batch_antibody['cdr_weights'] == cdr)
