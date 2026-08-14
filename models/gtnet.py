from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer, GCNConv
from torch_geometric.data import Batch, Data

from .gconv import  GCNConv2
from .gconv import GINConv as CustomGINConv, EELA

class SaveGrad:
    def __init__(self):
        self.grad = None
        self.activations = None

    def save_activation(self, activation):
        self.activations = activation.detach()

    def __call__(self, grad):
        self.grad = grad.detach()

def grad_cam(model, logits, target_layer, target_class):
    model.zero_grad()
    class_loss = logits[:, target_class].mean()
    class_loss.backward(retain_graph=True)
    activations = target_layer.activations
    gradients = target_layer.grad
    pooled_gradients = torch.mean(gradients, dim=0)
    for i in range(activations.size(1)):
        activations[:, i] *= pooled_gradients[i]
    node_contributions = torch.sum(activations, dim=1)
    node_contributions = F.relu(abs(node_contributions))
    raw_data = node_contributions
    node_contributions -= node_contributions.min()
    node_contributions /= node_contributions.max()
    return node_contributions, raw_data

class SimpleChannelAttention(nn.Module):
    def __init__(self, channel_dim):
        super(SimpleChannelAttention, self).__init__()
        self.channel_dim = channel_dim
        self.num_channels = 3
        self.weights = nn.Parameter(torch.ones(self.num_channels) / self.num_channels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        normalized_weights = self.softmax(self.weights)
        weighted_x = torch.zeros(x.size(0), self.channel_dim, device=x.device)
        for i in range(self.num_channels):
            weighted_x += normalized_weights[i] * x[:, :, i]
        return weighted_x, normalized_weights

class LGI_SUPER_Surv(torch.nn.Module):
    def __init__(self, gconv_dim, tlayer_dim, dataset_name, beta_value,
                 dataset_root=None, max_seq_len=None, in_dim=None, out_dim=None,
                 gconv_ffn_dropout=0., gconv_type='gin', num_layers=4,
                 middle_layer_type='none', readout='mean', input_dim=2227):
        super(LGI_SUPER_Surv, self).__init__()

        self.gconv_dim = gconv_dim
        self.tlayer_dim = tlayer_dim
        self.dataset_name = dataset_name
        self.in_dim = in_dim
        self.beta = beta_value
        self.readout = readout

        if readout == 'mean':
            self.readout_fn = global_mean_pool
        elif readout == 'add':
            self.readout_fn = global_add_pool

        self.node_encoder = nn.Linear(in_dim if in_dim else input_dim, self.gconv_dim)

        predict_head_modules = [nn.Linear(tlayer_dim, out_dim)]
        self.predict_head = nn.Sequential(*predict_head_modules)

        self.loc_norm_layer = nn.Linear(2, self.gconv_dim)
        self.gconvs = nn.ModuleList()
        self.middle_layers = nn.ModuleList()
        self.gconv_layers = num_layers

        for i in range(self.gconv_layers):
            if gconv_type == 'gin':
                self.gconvs.append(CustomGINConv(self.gconv_dim))
            elif gconv_type == 'gcn':
                self.gconvs.append(GCNConv2(self.gconv_dim, self.gconv_dim, self.beta))
            elif gconv_type == 'gen':
                self.gconvs.append(GENConv(self.gconv_dim, self.gconv_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer'))

        if middle_layer_type == 'residual':
            self.middle_layers.append(nn.BatchNorm1d(gconv_dim))
        elif middle_layer_type == 'mlp':
            self.middle_layers.append(nn.Sequential(
                nn.Dropout(gconv_ffn_dropout),
                nn.Linear(gconv_dim, gconv_dim),
                nn.LayerNorm(gconv_dim),
                nn.ReLU(),
                nn.Dropout(gconv_ffn_dropout)
            ))

        self.predict_head_grad = SaveGrad()
        self.pos_layer = nn.Sequential(nn.Linear(2, 1024), nn.Linear(1024, 1))

    def forward(self, **kwargs):
        data = kwargs['x_path']
        x, edge_index, batch, loc = data.x, data.edge_index, data.batch, data.centroid

        loc = self.loc_norm_layer(loc)
        x = self.node_encoder(x)

        for i in range(self.gconv_layers):
            x_t = x
            x, loc = self.gconvs[i](x, loc, edge_index, edge_weight=None, drop=0.2)
            x = x + x_t

        out = self.middle_layers[0](x)
        out = self.readout_fn(out, batch)
        out = out.mean(dim=0, keepdim=True)
        out = self.predict_head(out)

        logits = out
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None, None, None
