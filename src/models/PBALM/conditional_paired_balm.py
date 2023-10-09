# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from src.utils.data_utils import Alphabet
import torch
import torch.nn as nn
from .roberta import RobertaForMaskedLM, RobertaConfig


class PairedBALMWithStructuralAdatper(RobertaForMaskedLM):
    @classmethod
    def from_pretrained(cls, pretrained_path, cfg):
        balm_config = RobertaConfig.from_pretrained(pretrained_path)
        if isinstance(cfg.balm_config.adapter_layer_indices, int):
            cfg.balm_config.adapter_layer_indices = list(
                range(
                    0,
                    balm_config.num_hidden_layers,
                    cfg.balm_config.adapter_layer_indices,
                )
            )

        balm_config.update(cfg.balm_config)
        balm_config.adapter_config = deepcopy(balm_config)
        balm_config.adapter_config.update(cfg.adapter_config)

        pretrained_model = RobertaForMaskedLM.from_pretrained(
            pretrained_path, config=balm_config
        ).cpu()
        alphabet = Alphabet(name="paired_balm", featurizer="balm")
        model = cls(balm_config, alphabet)
        model.load_state_dict(
            pretrained_model.state_dict(), strict=False
        )

        del pretrained_model

        #  freeze pretrained parameters
        if cfg.balm_config.freeze:
            for pname, param in model.named_parameters():
                # flag = True
                # for l in cfg.balm_config.adapter_layer_indices:
                #     if str(l) in pname:
                #         flag = False
                if "crossattention" not in pname:
                    param.requires_grad = False
        return model

    def __init__(self, config, alphabet):
        super().__init__(config)
        self.config = config
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(
            self.config, "emb_layer_norm_before", False
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_ids: torch.Tensor,
        batch_antigen: dict,
    ):
        result = super().forward(
            input_ids=tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_antigen["feats"],
            encoder_attention_mask=batch_antigen["feats_mask"],
        )
        result = dict(
            logits=result.logits,
            representations=result.hidden_states,
            attentions=result.attentions,
        )
        return result
