from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

import math
from typing import Optional

import torch
from torch_geometric.utils import softmax
from torch_geometric.utils import degree,add_self_loops
import torch_scatter



class GCNConv0(gnn.MessagePassing):
    def __init__(self, emb_dim, edge_inut_dim):
        super().__init__(aggr='add')

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

        self.edge_encoder = nn.Linear(1, emb_dim)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()
        self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = x.to(self.linear.weight.dtype)
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr.unsqueeze(1))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding,norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GCNConvLSPE(gnn.MessagePassing):
    def __init__(self, emb_dim, edge_input_dim, pos_dim):
        super().__init__(aggr='add')

        self.node_encoder = nn.Linear(emb_dim, emb_dim)
        self.edge_encoder = nn.Linear(1,emb_dim)
        self.pos_encoder = nn.Linear(pos_dim, emb_dim)

        self.pos_update = nn.Linear(2 * emb_dim, emb_dim)

        self.combine_encodings = nn.Linear(2 * emb_dim, emb_dim)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.pos_encoder.reset_parameters()
        self.pos_update.reset_parameters()
        self.combine_encodings.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pos):

        x = self.node_encoder(x)
        edge_embedding = self.edge_encoder(edge_attr.unsqueeze(1))
        pos = self.pos_encoder(pos)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        p = self.update_positional_encodings(pos, edge_index, edge_embedding)

        combined = torch.cat([h, p], dim=-1)
        combined = self.combine_encodings(combined)

        return combined, p

    def message(self, x_j, edge_attr, norm):

        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update_positional_encodings(self, pos, edge_index, edge_embedding):

        row, col = edge_index
        pos_messages = torch.cat([pos[row], pos[col]], dim=-1)
        updated_pos = self.pos_update(pos_messages + edge_embedding)

        return updated_pos

    def update(self, aggr_out):

        return aggr_out

class GCNConv1(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, bias=1):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index,edge_weight):

        x = self.linear(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if edge_weight is not None:
            norm = deg_inv_sqrt[row] *edge_weight*deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] *deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class GCNConv2(gnn.MessagePassing):
    def __init__(self,  in_channels, out_channels,beta,bias=1):
        super().__init__(aggr='add')
        self.x_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.loc_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.beta = beta

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.x_linear.reset_parameters()
        self.loc_linear.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x,loc, edge_index,edge_weight,drop):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.x_linear(x)
        loc = self.loc_linear(loc)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if edge_weight is not None:
            norm = deg_inv_sqrt[row] *edge_weight*deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] *deg_inv_sqrt[col]
        out_x = self.propagate(edge_index, x=x,loc=loc,norm=norm)
        out_loc = self.propagate(edge_index, x=None,loc=loc,norm=norm)

        if self.bias is not None:
            out_x += self.bias

        out_x = F.relu(out_x)
        out_loc = torch.tanh(out_loc)
        return out_x,out_loc

    def message(self, x_j, loc_j,norm):
        if x_j is not None:
            mes = self.beta*loc_j+x_j
        else:
            mes = loc_j
        mes_norm =  mes / (torch.norm(mes, dim=1, keepdim=True) + 1e-6)
        return norm.view(-1, 1) * mes_norm

    def update(self, aggr_out):
        return aggr_out

