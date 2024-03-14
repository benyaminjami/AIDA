import torch
from torch import nn

from src.models.protein_mpnn_cmlm.protein_mpnn import ProteinMPNNCMLM

from .BALM.conditional_balm import BALMWithStructuralAdatper
from .PBALM.conditional_paired_balm import PairedBALMWithStructuralAdatper


class MPNNcBALM(nn.Module):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        super().__init__()
        self.encoder = ProteinMPNNCMLM.from_pretrained(self.cfg.encoder)
        if cfg.decoder.model == "paired_balm":
            self.decoder = PairedBALMWithStructuralAdatper.from_pretrained(
                self.cfg.decoder.pretrained_path, cfg.decoder
            )
        elif cfg.decoder.model == "balm":
            self.decoder = BALMWithStructuralAdatper.from_pretrained(
                self.cfg.decoder.pretrained_path, cfg.decoder
            )
        else:
            raise NotImplementedError
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward(self, batch, **kwargs):
        batch_antigen = batch["antigen"]
        batch_antibody = batch["antibody"]
        # encoder_logits, encoder_out = self.encoder(batch_antigen, return_feats=True, **kwargs)
        batch_antigen = self.forward_encoder(batch_antigen)

        esm_logits = self.forward_decoder(batch_antibody, batch_antigen)

        return esm_logits

    def forward_encoder(self, batch_antigen):
        encoder_logits, encoder_out = self.encoder(batch_antigen, return_feats=True)

        encoder_out["logits"] = encoder_logits
        encoder_out["coord_mask"] = batch_antigen["coord_mask"]
        batch_antigen.update(encoder_out)

        if self.cfg.get("epitope_selection", "topk") == "topk":
            _, indices = torch.topk(
                batch_antigen["dists"],
                k=min(48, batch_antigen["dists"].size(1)),
                dim=1,
                largest=False,
            )
            epitope_mask = torch.zeros_like(batch_antigen["dists"], dtype=torch.bool)
            rows = torch.arange(indices.size(0)).unsqueeze(-1)
            epitope_mask[rows, indices] = True
        else:
            raise NotImplementedError

        if self.cfg.get("epitope_handling", "cutting") == "cutting":
            batch_antigen["feats_mask"] = batch_antigen["coord_mask"][epitope_mask].view(
                (epitope_mask.size(0), -1)
            )
            batch_antigen["feats"] = batch_antigen["feats"][epitope_mask].view(
                (epitope_mask.size(0), -1, 256)
            )
        elif self.cfg.get("epitope_handling") == "masking":
            batch_antigen["feats_mask"] = batch_antigen["coord_mask"] & epitope_mask
        else:
            raise NotImplementedError

        return batch_antigen

    def forward_decoder(self, batch_antibody, batch_antigen, need_attn_weights=False):
        esm_logits = self.decoder(
            tokens=batch_antibody["prev_tokens"],
            attention_mask=batch_antibody["attention_mask"],
            position_ids=batch_antibody["position_ids"],
            chain_ids=batch_antibody["chain_ids"],
            batch_antigen=batch_antigen,
        )["logits"]

        return esm_logits
