import torch
from torch import nn
original_torch = torch
original_module = nn.Module
from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.models.esm import EvolutionaryScaleModeling as ESM
nn.Module = original_module
torch = original_torch
import os
from torchdrug.models import GeometryAwareRelationalGraphNeuralNetwork
from collections.abc import Sequence
from pathlib import Path
from torch.nn import functional as F
from torch_scatter import scatter_add


class ESMGearNet(nn.Module):
    @classmethod
    def from_pretrained(cls, cfg, ckpt=None):
        # return None
        if cfg.get("fuse_esm", True):
            ckpt = "siamdiff_esm_gearnet.pth"
        else:
            ckpt = "mc_gearnet_edge.pth"
        ckpt_path = Path(cfg.pretrained_path, ckpt)
        model = cls(cfg.get("fuse_esm", True), use_adapter=cfg.get("use_adapter", False))
        if os.path.exists(str(ckpt_path)):
            state_dict = torch.load(str(ckpt_path))
            if cfg.get("fuse_esm", True):
                model.load_state_dict(state_dict, strict=True)
            else:
                model.structure_model.load_state_dict(state_dict, strict=True)
        if hasattr(model, "sequence_model"):
            for param in model.sequence_model.parameters():
                param.requires_grad = False
        for param in model.structure_model.parameters():
            param.requires_grad = False

        return model

    def __init__(self, fuse_esm=True, proj_dropout=0.2, use_adapter=False):
        super(ESMGearNet, self).__init__()
        self.fuse_esm = fuse_esm
        self.output_dim = 0
        if fuse_esm:
            esm = ESM(
                path="~/scrach/Conditional-BALM/data/pretrained_models/pretrained-ESM/",
                model="ESM-1b",
            )
            self.sequence_model = esm
            self.output_dim = self.sequence_model.output_dim

        gearnet = GearNet(
            input_dim=1280 if fuse_esm else 21,
            hidden_dims=[512, 512, 512, 512, 512, 512],
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
            num_relation=7,
            edge_input_dim=59 if not fuse_esm else None,
            num_angle_bin=8 if not fuse_esm else None,
            # use_adapter=use_adapter
        )
        self.structure_model = gearnet
        self.output_dim += self.structure_model.output_dim

        self.output_proj = nn.Linear(self.output_dim, 256, bias=False)

        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, batch):
        if not self.fuse_esm:
            return self.forward_gearnet(batch)
        graph = batch["graphs"]
        output1 = self.sequence_model(graph, graph.residue_feature.float())
        node_output1 = output1.get("node_feature", output1.get("residue_feature"))
        output2 = self.structure_model(graph, node_output1)
        node_output2 = output2.get("node_feature", output2.get("residue_feature"))

        node_feature = torch.cat([node_output1, node_output2], dim=-1)
        node_feature = self.output_proj(node_feature)
        graph_feature = torch.cat(
            [output1["graph_feature"], output2["graph_feature"]], dim=-1
        )
        # Initialize the 3D tensor with zeros
        reshaped_node_feature = torch.zeros(
            batch["graphs"].batch_size, batch["mask"].size(1), node_feature.size(1)
        ).to(node_feature.device)
        # Use the mask to fill values from the 2D tensor into the 3D tensor
        for i in range(batch["graphs"].batch_size):
            reshaped_node_feature[i, batch["mask"][i]] = node_feature[
                batch["mask"][:i].sum(): batch["mask"][: i + 1].sum()
            ]

        return None, {"graph_feature": graph_feature, "feats": reshaped_node_feature}

    def forward_gearnet(self, batch):
        graph = batch["graphs"]
        output = self.structure_model(graph, graph.residue_feature.float())
        node_feature = output.get("node_feature", output.get("residue_feature"))

        node_feature = self.output_proj(node_feature)
        node_feature = self.proj_dropout(node_feature) 
        graph_feature = output["graph_feature"]

        reshaped_node_feature = torch.zeros(
            batch["graphs"].batch_size, batch["mask"].size(1), node_feature.size(1)
        ).to(node_feature.device)
        # Use the mask to fill values from the 2D tensor into the 3D tensor
        for i in range(batch["graphs"].batch_size):
            reshaped_node_feature[i, batch["mask"][i]] = node_feature[
                batch["mask"][:i].sum(): batch["mask"][: i + 1].sum()
            ]

        return None, {"graph_feature": graph_feature, "feats": reshaped_node_feature}


class GearNet(nn.Module, core.Configurable):
    """
    Geometry Aware Relational Graph Neural Network proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        edge_input_dim=None,
        num_angle_bin=None,
        short_cut=False,
        batch_norm=False,
        activation="relu",
        concat_hidden=False,
        readout="sum",
        dropout=0.1,
        layer_norm=False,
        use_adapter=False,
    ):
        super(GearNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm
        self.use_adapter = use_adapter

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeometricRelationalGraphConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    None,
                    batch_norm,
                    activation,
                )
            )
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(
                    layers.GeometricRelationalGraphConv(
                        self.edge_dims[i],
                        self.edge_dims[i + 1],
                        num_angle_bin,
                        None,
                        batch_norm,
                        activation,
                    )
                )

        if use_adapter:
            self.edge_adapter1 = nn.ModuleList()
            self.edge_adapter2 = nn.ModuleList()
            for i in range(1, len(self.edge_dims) - 1):
                self.edge_adapter1.append(nn.Linear(self.edge_dims[i], self.edge_dims[i]//4))
                self.edge_adapter2.append(nn.Linear(self.edge_dims[i]//4, self.edge_dims[i]))

            self.node_adapter1 = nn.ModuleList()
            self.node_adapter2 = nn.ModuleList()
            for i in range(1, len(self.dims) - 1):
                self.node_adapter1.append(nn.Linear(self.dims[i], self.dims[i]//4))
                self.node_adapter2.append(nn.Linear(self.dims[i]//4, self.dims[i]))

        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))
        elif batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # self.dropout = nn.Dropout(dropout)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)

            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = (
                    graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                )
                update = scatter_add(
                    edge_hidden * edge_weight,
                    node_out,
                    dim=0,
                    dim_size=graph.num_node * self.num_relation,
                )
                update = update.view(
                    graph.num_node, self.num_relation * edge_hidden.shape[1]
                )
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                if self.use_adapter:
                    edge_hidden = edge_hidden + self.edge_adapter2(F.relu(self.edge_adapter1(edge_hidden)))
                edge_input = edge_hidden
            # hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if hasattr(self, "layer_norms"):
                hidden = self.layer_norms[i](hidden)
            elif hasattr(self, "batch_norms"):
                hidden = self.batch_norms[i](hidden)
            if self.use_adapter:
                hidden = hidden + self.node_adapter2(F.relu(self.edge_adapter1(hidden)))
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {"graph_feature": graph_feature, "node_feature": node_feature}


class FusionNetwork(nn.Module, core.Configurable):
    @classmethod
    def from_pretrained(cls, cfg, ckpt=None):
        # return None
        if cfg.get("fuse_esm", True):
            ckpt = "siamdiff_esm_gearnet.pth"
        else:
            ckpt = "mc_gearnet_edge.pth"
        ckpt_path = Path(cfg.pretrained_path, ckpt)
        model = cls(cfg.get("fuse_esm", True), use_adapter=cfg.get("use_adapter", False))
        if os.path.exists(str(ckpt_path)):
            state_dict = torch.load(str(ckpt_path))
            if cfg.get("fuse_esm", True):
                model.load_state_dict(state_dict, strict=True)
            else:
                model.structure_model.load_state_dict(state_dict, strict=True)
        if hasattr(model, "sequence_model"):
            for param in model.sequence_model.parameters():
                param.requires_grad = False
        for param in model.structure_model.parameters():
            param.requires_grad = False

        return model

    def __init__(self, fusion="series", cross_dim=None, proj_dropout=0.2):
        super(FusionNetwork, self).__init__()
        esm = ESM(
                path="~/scrach/Conditional-BALM/data/pretrained_models/pretrained-ESM/",
                model="ESM-1b",
            )
        self.sequence_model = esm
        gearnet = GearNet(
            input_dim=1280,
            hidden_dims=[512, 512, 512, 512, 512, 512],
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
            num_relation=7,
        )
        self.structure_model = gearnet
        self.fusion = fusion
        if fusion in ["series", "parallel"]:
            self.output_dim = self.sequence_model.output_dim + self.structure_model.output_dim     
        else:
            raise ValueError("Not support fusion scheme %s" % fusion)
        self.output_proj = nn.Linear(self.output_dim, 256, bias=False)

        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, graph, input, all_loss=None, metric=None):
        # Sequence model
        output1 = self.sequence_model(graph, input, all_loss, metric)
        node_output1 = output1.get("node_feature", output1.get("residue_feature"))
        # Structure model
        if self.fusion == "series":
            input = node_output1
        output2 = self.structure_model(graph, input, all_loss, metric)
        node_output2 = output2.get("node_feature", output2.get("residue_feature"))
        # Fusion
        if self.fusion in ["series", "parallel"]:
            node_feature = torch.cat([node_output1, node_output2], dim=-1)
            graph_feature = torch.cat([
                output1["graph_feature"],
                output2["graph_feature"]
            ], dim=-1)
        

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }