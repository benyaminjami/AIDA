# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn as nn

from src.utils.data_utils import Alphabet

from .modeling_balm import BALMForMaskedLM, EsmConfig


class BALMWithStructuralAdatper(BALMForMaskedLM):
    @classmethod
    def from_pretrained(cls, pretrained_path, cfg):
        balm_config = EsmConfig.from_pretrained(pretrained_path)
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

        pretrained_model = BALMForMaskedLM.from_pretrained(pretrained_path, config=balm_config)
        alphabet = Alphabet(name="esm", featurizer="balm")
        model = cls(balm_config, alphabet)
        model.load_state_dict(
            pretrained_model.state_dict(), strict=False
        )  # TODO change this later

        del pretrained_model

        # freeze pretrained parameters
        for pname, param in model.named_parameters():
            if "crossattention" not in pname and "chain_embedding" not in pname:
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
        self.emb_layer_norm_before = getattr(self.config, "emb_layer_norm_before", False)
        self._init_chain_embedding()

    def _init_chain_embedding(self):
        self.chain_embedding = nn.Embedding(
            3, self.config.hidden_size, padding_idx=0  # 0 = Pad | 1 = Light | 2 = Heavy
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_ids: torch.Tensor,
        batch_antigen: dict,
    ):
        embedding_output = self.esm.embeddings(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        embedding_output = embedding_output + self.chain_embedding(chain_ids)
        embedding_output = embedding_output * attention_mask.unsqueeze(-1).to(
            embedding_output.dtype
        )
        result = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=embedding_output,
            encoder_hidden_states=batch_antigen["feats"],
            encoder_attention_mask=batch_antigen["feats_mask"],
        )
        result = dict(
            logits=result.logits,
            representations=result.hidden_states,
            attentions=result.attentions,
        )
        return result
