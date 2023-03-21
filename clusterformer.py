import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from nodeformer_cluster import *
import time


class Clusterformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True,
                 use_gumbel=True, use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=1,
                 rb_trans='sigmoid', use_edge_loss=True, tau=0.25):
        super().__init__()
        self.rb_order = rb_order
        self.hidden_channels = hidden_channels * num_layers + hidden_channels if use_jk else hidden_channels
        print(self.hidden_channels)
        self.tau = tau
        self.v_layer = 1
        self.use_edge_loss = use_edge_loss
        self.use_jk = use_jk
        self.num_layers = num_layers

        # encoder & decoder
        self.encoders = nn.ModuleList()
        for i in range(self.v_layer + 1):
            in_dim = in_channels if i == 0 else hidden_channels
            self.encoders.append(ClusterNodeFormer(in_dim, hidden_channels, num_layers=num_layers,
                                                   dropout=dropout, num_heads=num_heads, use_bn=use_bn,
                                                   nb_random_features=nb_random_features, use_gumbel=use_gumbel,
                                                   use_residual=use_residual, use_act=use_act, use_jk=use_jk,
                                                   nb_gumbel_sample=nb_gumbel_sample, rb_order=rb_order,
                                                   rb_trans=rb_trans))
        self.fc = nn.Linear(self.hidden_channels, hidden_channels) if use_jk else nn.Linear(hidden_channels,
                                                                                            hidden_channels)
        self.activation = nn.ELU()

        self.decoder = nn.Linear(self.hidden_channels * (self.v_layer + 1), out_channels)

    def forward(self, cluster_loader, v_adjs, device="cpu"):
        # rb_order is always 1
        # TODO consider num of vnodes < num_parts
        N, C, H = cluster_loader.cluster_data.data.x.size(0), self.v_layer + 1, self.hidden_channels
        x = torch.zeros((N, C, H)).to(device)  # (N, 2, H)

        n_clsts = len(cluster_loader)
        v_nodes = torch.zeros((n_clsts, H)).float().to(device)
        link_losses = torch.zeros((n_clsts + 1, self.num_layers)).float().to(device)  # link_loss of clusters
        n_per_clst = torch.zeros((n_clsts + 1, 1)).long().to(device)  # to get weighted link_losses of clusters
        scatter_idx = []
        for i, sampled_data in enumerate(cluster_loader):
            # get nodes' feat (contain v_node)
            x_i = sampled_data.x
            n_nodes_i = x_i.size(0)
            v_node_i = torch.mean(x_i, dim=0, keepdim=True)  # TODO better v_node feat
            x_extend_i = torch.cat([x_i, v_node_i], dim=0).to(device)
            n_per_clst[i] = n_nodes_i
            # get edge_index (contain v_nodes)
            adjs_i = []
            edge_index_i = sampled_data.edge_index

            idx_list_i = torch.arange(n_nodes_i)[None, :]
            v_edges = torch.cat([torch.ones(n_nodes_i, )[None, :] * n_nodes_i, idx_list_i], dim=0).long()
            edge_index_i = torch.cat([edge_index_i, v_edges], dim=1)
            v_edges[[0, 1]] = v_edges[[1, 0]]
            edge_index_i = torch.cat([edge_index_i, v_edges], dim=1)
            adjs_i.append(edge_index_i.to(device))

            # encode
            out_i, link_loss_i_ = self.encoders[0](x_extend_i, adjs_i, self.tau)
            link_losses[i] = link_loss_i_
            v_nodes[i] = out_i[-1]
            x[sampled_data.n_clst_id, 0, :] = out_i[:-1, :]
            scatter_idx.append(sampled_data.n_clst_id)

        # v_node
        v_nodes = self.activation(self.fc(v_nodes))
        v_out, v_link_loss_ = self.encoders[1](v_nodes, v_adjs, self.tau)  # TODO edge weight & vnode
        link_losses[-1] = v_link_loss_  # (N_clst + 1, L)
        n_per_clst[-1] = v_nodes.size(0)  # (N_clst + 1, 1)

        # aggregate
        # TODO how to calc link loss better
        for i in range(v_out.size(0)):
            x[scatter_idx[i], 1, :] = v_out[0]
        link_loss_ = torch.sum(link_losses * n_per_clst / n_per_clst.sum(), dim=0, keepdim=False)  # (L,)

        out = self.decoder(x.reshape(N, -1))

        if self.use_edge_loss:
            return out, link_loss_
        return out

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.fc.reset_parameters()
        self.decoder.reset_parameters()
