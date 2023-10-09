from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.generator import sample_from_categorical
from src.models.protein_mpnn_cmlm import FixedBackboneDesignEncoderDecoder

from .decoder import MPNNSequenceDecoder
from .encoder import MPNNEncoder


@dataclass
class ProteinMPNNConfig:
    n_enc_layers: 3
    d_model: int = 128
    d_node_feats: int = 128
    d_edge_feats: int = 128
    k_neighbors: int = 48
    augment_eps: float = 0.0
    n_enc_layers: int = 3
    dropout: float = 0.1

    encoder_only: bool = False

    # decoder-only
    n_dec_layers: 3
    n_vocab: int = 22
    n_dec_layers: int = 3
    random_decoding_order: bool = True
    nar: bool = True
    crf: bool = False
    use_esm_alphabet: bool = False


class ProteinMPNNCMLM(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ProteinMPNNConfig()

    @classmethod
    def from_pretrained(cls, cfg, ckpt="best.ckpt"):
        ckpt_path = Path(cfg.pretrained_path, "checkpoints", ckpt)
        model_cfg = deepcopy(cfg)
        del model_cfg["pretrained_path"]
        model = cls(model_cfg)
        state_dict = torch.load(str(ckpt_path))["state_dict"]
        state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
        state_dict.pop("decoder.out_proj.weight")
        state_dict.pop("decoder.out_proj.bias")
        model.load_state_dict(state_dict, strict=False)
        return model

    def __init__(self, cfg=OmegaConf.create()) -> None:
        super().__init__(cfg)

        self.encoder = MPNNEncoder(
            node_features=self.cfg.d_node_feats,
            edge_features=self.cfg.d_edge_feats,
            hidden_dim=self.cfg.d_model,
            num_encoder_layers=self.cfg.n_enc_layers,
            k_neighbors=self.cfg.k_neighbors,
            augment_eps=self.cfg.augment_eps,
            dropout=self.cfg.dropout,
            n_vocab=self.cfg.n_vocab,
            encoder_only=self.cfg.encoder_only,
        )

        alphabet = None
        self.padding_idx = 0
        self.mask_idx = 1
        self.encoder_only = self.cfg.encoder_only

        self.decoder = MPNNSequenceDecoder(
            n_vocab=self.cfg.n_vocab,
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_dec_layers,
            random_decoding_order=self.cfg.random_decoding_order,
            dropout=self.cfg.dropout,
            nar=self.cfg.nar,
            crf=self.cfg.crf,
            alphabet=alphabet,
        )

    def _forward(
        self,
        coords,
        coord_mask,
        prev_tokens,
        token_padding_mask=None,
        target_tokens=None,
        return_feats=False,
        **kwargs
    ):
        coord_mask = coord_mask.float()
        encoder_out = self.encoder(X=coords, mask=coord_mask)

        logits, feats = self.decoder(
            prev_tokens=prev_tokens,
            memory=encoder_out,
            memory_mask=coord_mask,
            target_tokens=target_tokens,
            **kwargs
        )

        if return_feats:
            return logits, feats
        return logits

    def forward(self, batch, return_feats=True, **kwargs):
        coord_mask = batch["coord_mask"].float()

        encoder_out = self.encoder(
            X=batch["coords"],
            mask=coord_mask,
            residue_idx=batch.get("residue_idx", None),
            chain_idx=batch.get("chain_idx", None),
            prev_tokens=batch["tokens"],
        )
        if self.encoder_only:
            return None, {"feats": encoder_out["node_feats"]}
        logits, feats = self.decoder(
            prev_tokens=batch["tokens"],
            memory=encoder_out,
            memory_mask=coord_mask,
            target_tokens=batch.get("tokens"),
            **kwargs
        )
        feats["feats"] = torch.concat((feats["feats"], encoder_out["node_feats"]), dim=-1)
        if return_feats:
            return logits, feats
        return logits
