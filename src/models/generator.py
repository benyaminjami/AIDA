import torch

from src.utils.data_utils import Alphabet


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = ((output_masks.sum(1, keepdim=True).type_as(output_scores)) * p).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _random_unmasking(output_scores, output_masks, p):
    mask = torch.zeros_like(output_scores).type(torch.bool)
    for i in range(output_scores.size(0)):
        probs = torch.zeros_like(output_masks[i]).type(torch.float)
        probs[output_masks[i]] = -(output_scores[i][output_masks[i]])
        ind = torch.multinomial(probs, max(1, int(output_masks[i].sum().item()*0.2)))
        mask[i][ind] = True
    return mask


def exists(obj):
    return obj is not None


def new_arange(x, *size):
    """Return a Tensor of `size` filled with a range function on the device of x.

    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def maybe_remove_batch_dim(tensor):
    if len(tensor.shape) > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


class IterativeRefinementGenerator:
    def __init__(
        self,
        alphabet: Alphabet = None,
        max_iter: int = 1,
        strategy: str = "denoise",
        temperature: float = None,
        **kwargs
    ):
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self.max_iter = max_iter
        self.strategy = strategy
        self.temperature = temperature

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
        strategy = strategy or self.strategy
        temperature = temperature or self.temperature

        batch_antibody = batch["antibody"]
        batch_antigen = batch["antigen"]
        # 0) encoding
        encoder_out = model.forward_encoder(batch_antigen)

        # 1) initialized from all mask tokens
        initial_output_tokens = batch_antibody["prev_tokens"]
        initial_output_tokens_mask = batch_antibody['prev_token_mask']
        prev_scores = torch.zeros_like(initial_output_tokens)
        decoder_info = dict(
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
        )

        if need_attn_weights:
            attns = []  # list of {'in', 'out', 'attn'} for all iteration

        # iterative refinement
        for step in range(max_iter):
            # 2.1: predict
            decoder_out = model.forward_decoder(
                batch_antibody=batch_antibody,
                batch_antigen=encoder_out,
                need_attn_weights=need_attn_weights,
            )
            output_tokens, output_scores = sample_from_categorical(
                decoder_out,
                temperature=temperature if step < max_iter-1 else 0,
                prev_tokens=batch_antibody['prev_tokens'],
                mask=initial_output_tokens_mask
            )
            

            # 2.3: update
            if replace_visible_tokens:
                visible_token_mask = ~batch_antibody["prev_token_mask"]
                visible_tokens = batch_antibody["prev_tokens"]
                output_tokens = torch.where(visible_token_mask, visible_tokens, output_tokens)
                output_scores = torch.where(visible_token_mask, prev_scores, output_scores)

            # 2.2: re-mask skeptical parts of low confidence
            # skeptical decoding (depend on the maximum decoding steps.)
            # skeptical_mask = _skeptical_unmasking(
            #     output_scores=output_scores,
            #     output_masks=initial_output_tokens_mask,
            #     p=1 - (step + 1) / max_iter,
            # )
            # output_tokens[~(~skeptical_mask * initial_output_tokens_mask)] = batch_antibody['prev_tokens'][~(~skeptical_mask * initial_output_tokens_mask)]
            if strategy == "mask_predict" and (step + 1) < max_iter:
                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=initial_output_tokens_mask,
                    p=1 - (step + 1) / max_iter,
                )
                output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
                output_scores.masked_fill_(skeptical_mask, 0.0)
                batch_antibody["prev_token_mask"] = skeptical_mask

            elif strategy == "random_mask_predict" and (step + 1) < max_iter:
                random_mask = _random_unmasking(
                    output_scores=output_scores,
                    output_masks=initial_output_tokens_mask,
                    p=0.5,
                )
                output_tokens.masked_fill_(random_mask, self.mask_idx)
                output_scores.masked_fill_(random_mask, 0.0)
                batch_antibody["prev_token_mask"] = random_mask
            decoder_info["history"].append(output_tokens.clone())
            batch_antibody["prev_tokens"] = output_tokens
            prev_scores = output_scores
            # if need_attn_weights:
            #     attns.append(
            #         dict(
            #             input=maybe_remove_batch_dim(prev_decoder_out["output_tokens"]),
            #             output=maybe_remove_batch_dim(output_tokens),
            #             attn_weights=maybe_remove_batch_dim(decoder_out["attentions"]),
            #         )
            #     ) TODO: add attention weights

            decoder_info.update(step=step + 1)
        batch_antibody["prev_token_mask"] = initial_output_tokens_mask
        if need_attn_weights:
            return output_tokens, output_scores, attns
        return output_tokens, output_scores, decoder_info['history']


def sample_from_categorical(logits=None, temperature=1.0, prev_tokens=None, mask=None):
    if temperature:
        dist = torch.distributions.Categorical(probs=logits.softmax(-1).div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    elif temperature is None or temperature == 0.0:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
