import torch
import torch.nn as nn

from .features import (
    PositionWiseFeedForward,
    ProteinFeatures,
    cat_neighbors_nodes,
    gather_nodes,
)


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer."""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class MPNNEncoder(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        n_vocab=23,
        num_encoder_layers=3,
        k_neighbors=64,
        augment_eps=0.05,
        dropout=0.1,
        encoder_only=False,
    ):
        super().__init__()
        # Hyperparameters
        self.encoder_only = encoder_only
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        if self.encoder_only:
            self.W_out = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
            self.token_embed = nn.Embedding(n_vocab, hidden_dim)
        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def featurize(self, X, mask, residue_idx=None, chain_encoding_all=None):
        device = X.device
        bsz, n_nodes = X.shape[0], X.shape[1]

        if residue_idx is None:
            residue_idx = torch.arange(0, n_nodes)[None, :].repeat(bsz, 1).to(device)
        if chain_encoding_all is None:
            chain_encoding_all = torch.ones((bsz, n_nodes)).to(device)

        E, E_idx = self.features(X, mask, residue_idx=residue_idx, chain_labels=chain_encoding_all)

        return E, E_idx

    def forward(self, X, mask, prev_tokens, residue_idx=None, chain_idx=None):
        """

        Returns: dict of
            node_feats: [bsz, n_nodes, d]
            edge_feats: [bsz, n_nodes, n_edges, d]
            edge_idx: [bsz, n_nodes, n_edges]
        """
        # 1. prepare edge features for protein
        E, E_idx = self.featurize(X, mask, residue_idx=residue_idx, chain_encoding_all=chain_idx)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        if not self.encoder_only:
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        else:
            h_V = self.token_embed(prev_tokens)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        if self.encoder_only:
            h_V = self.W_out(h_V)
        return {"node_feats": h_V, "edge_feats": h_E, "edge_idx": E_idx}
