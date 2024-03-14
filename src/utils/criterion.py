import torch
from torch import Tensor, nn
from torch.nn import functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, weights=None) -> Tensor:
        """
        scores: [N, ..., C], unnormalized scores
        target: [N, ...]
        mask: [N, ...], where elements with `True` are allowed and `False` are masked-out
        """

        bsz, _ = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights

        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
            # label_mask = label_mask.float()
            # sample_size = label_mask.sum(-1)  # sample size should be set to valid coordinates
            # loss = (loss * label_mask).sum(-1) / sample_size
            # loss = loss.sum() / sample_size.size(0)
            # nll_loss = (nll_loss * label_mask).sum(-1) / sample_size
            # nll_loss = nll_loss.sum() / sample_size.size(0
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        ppl = torch.exp(nll_loss)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ppl": ppl.data,
            "fullseq_loss": fullseq_loss.data,
            "fullseq_nll_loss": fullseq_nll_loss.data,
            "bsz": bsz,
            "sample_size": sample_size,
            "sample_ratio": sample_size / n_tokens,
            "nonpad_ratio": n_nonpad_tokens / n_tokens,
        }
        return loss, logging_output


def focal_loss(probs, target, ignore_index=None, reduce=True, mu=2):
    flag = False
    if target.dim() == probs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -torch.log(probs.gather(dim=-1, index=target))
    p_t = 1 - probs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        p_t.masked_fill_(pad_mask, 0.0)

    loss = (p_t**mu) * nll_loss

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        loss = loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        loss = loss.sum()

    return loss, nll_loss


class FocalLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, weights=None) -> Tensor:
        """
        scores: [N, ..., C], unnormalized scores
        target: [N, ...]
        mask: [N, ...], where elements with `True` are allowed and `False` are masked-out
        """

        bsz, _ = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        loss, nll_loss = focal_loss(
            probs=F.softmax(scores, dim=-1),
            target=target,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights

        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        ppl = torch.exp(nll_loss)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ppl": ppl.data,
            "fullseq_loss": fullseq_loss.data,
            "fullseq_nll_loss": fullseq_nll_loss.data,
            "bsz": bsz,
            "sample_size": sample_size,
            "sample_ratio": sample_size / n_tokens,
            "nonpad_ratio": n_nonpad_tokens / n_tokens,
        }
        return loss, logging_output
