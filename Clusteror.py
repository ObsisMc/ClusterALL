import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

import tqdm
import math
import os


class Clusteror(nn.Module):
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, num_layers, use_jk=False,
                 dropout=0.1, num_parts=10):
        super().__init__()

        self.dropout = dropout
        self.num_parts = num_parts
        self.encoder = encoder

        self.fcs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        # encode
        jk_channels = hidden_channels * num_layers + hidden_channels
        self.fcs["proj_in"] = nn.Linear(in_channels, in_channels)
        self.bns["ln_in"] = nn.LayerNorm(in_channels)
        self.bns["ln_encode"] = nn.LayerNorm(jk_channels if use_jk else hidden_channels)
        self.activations["elu"] = nn.ELU()

        # aggregate vnodes' feat
        self.fcs["aggr"] = nn.Linear(jk_channels * 2, jk_channels) if use_jk else nn.Linear(hidden_channels * 2,
                                                                                            hidden_channels)

        # decode
        self.fcs["output"] = nn.Linear(jk_channels if use_jk else hidden_channels, out_channels)

        # vnodes
        self.vnode_embed = nn.Parameter(torch.randn(self.num_parts, in_channels))  # virtual node

    def forward(self, x, **kwargs):
        # set clusters explicit
        # out, loss = self.foward_cluster(x, **kwargs)

        # end to end: get embedding and find clusters
        out, link_loss, infos = self.forward_fc(x, **kwargs)
        return out, link_loss, infos

    def forward_fc(self, x, **kwargs):
        adjs, tau = kwargs["adjs"], getattr(kwargs, "tau", 0.25)
        x[-self.num_parts:] = self.vnode_embed

        # encode
        x, loss, weight = self.encoder(x, adjs=adjs, tau=tau)

        # cluster
        weight_final = weight[0]
        n = x.size(0) - self.num_parts
        v2n_w = weight_final[-n * self.num_parts:].reshape(n, -1)  # (N,K)
        cluster_idx = torch.argmax(v2n_w, dim=1, keepdim=False)
        cluster_idx += n

        # aggr
        v_x = x[-self.num_parts:]
        n_x = torch.cat([x[:-self.num_parts], x[cluster_idx]], dim=1)
        n_x = self.activations["elu"](self.bns["ln_encode"](self.fcs["aggr"](n_x)))
        n_x = F.dropout(n_x, p=self.dropout, training=self.training)

        # decode
        x = torch.cat([n_x, v_x])
        x = self.fcs["output"](x)
        out = x[:-self.num_parts]

        # interpretability
        cluster_reps = x[:-self.num_parts]
        cluster_mapping = cluster_idx

        return out, loss, (cluster_reps, cluster_mapping)

    def forward_cluster(self, x, **kwargs):
        mappings = kwargs["mappings"]
        n2v_mapping, v_mapping = mappings["n2v"], mappings["vs2vg"]
        num_vnodes = len(set(n2v_mapping.tolist()))

        x = self.activations["elu"](self.bns["ln_in"](self.fcs["proj_in"](x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x[-num_vnodes:] = self.vnode_embed[v_mapping]

        # encode
        # print("before encode", x[-num_vnodes-2:])
        x, loss = self.encode(x, **kwargs)

        # if torch.any(torch.isnan(x)):
        #     print(x)
        #     raise Exception

        x = self.activations["elu"](self.bns["ln_encode"](x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # decode
        # map nodes to vnodes and aggr
        x = torch.cat([x, x[n2v_mapping]], dim=1)
        x = self.activations["elu"](self.bns["ln_encode"](self.fcs["aggr"](x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if torch.any(torch.isnan(x)):
        #     print(x)
        #     raise Exception

        # decode
        out = self.fcs["output"](x)

        out = out[:-num_vnodes]

        return out, loss

    def encode(self, x, **kwargs):
        adjs, tau = kwargs["adjs"], getattr(kwargs, "tau", 0.25)
        return self.encoder(x, adjs, tau=tau)

    def reset_parameters(self, init_feat: torch.Tensor):
        self.encoder.reset_parameters()
        for key, item in self.fcs.items():
            self.fcs[key].reset_parameters()
        for key, item in self.bns.items():
            self.bns[key].reset_parameters()
        self.vnode_embed = nn.Parameter(init_feat)


class MyDataLoader:
    def __init__(self, data: Data, num_parts, batch_size: int = -1, loader=None, shuffle=False):
        if loader is not None:
            raise NotImplemented("Customized loader isn't implemented")

        self.data_aug, self.n_ids, self.mapping, self.vnode_init, self.v_gmap = self.__pre_process(data,
                                                                                                   num_parts=num_parts)
        self.num_parts = num_parts

        self.num_n = len(self.n_ids)
        self.num_n_aug = self.data_aug.x.size(0)
        self.batch_size = self.num_n if batch_size == -1 else batch_size
        self.loader = loader
        self.loader_len = math.ceil(self.num_n / self.batch_size)

        self.shuffle = shuffle
        self.batch_nid = self.n_ids
        if self.shuffle:
            self.batch_nid = self.batch_nid[torch.randperm(self.num_n)]

    def __len__(self):
        return self.loader_len

    def __getitem__(self, idx):
        # original nodes
        if idx >= len(self):
            raise StopIteration

        s_o, e_o = self.batch_size * idx, self.batch_size * (idx + 1)
        nids = self.batch_nid[s_o:e_o]  # pay attention to the last batch

        # find vnodes
        o2v_mapping = [self.mapping[nid.item()] for nid in nids]
        vids = torch.tensor(sorted(list(set(o2v_mapping)))).long()
        o2v_mapping += vids.tolist()

        # add vnodes
        idx_aug = torch.cat([nids, vids], dim=0)
        sampled_x = self.data_aug.x[idx_aug]
        sampled_y = self.data_aug.y[nids] if self.data_aug.y is not None else None  # only need original label
        sampled_edge_index, _ = subgraph(idx_aug, self.data_aug.edge_index, num_nodes=self.num_n_aug,
                                         relabel_nodes=True)

        # mapping in subgraph
        vid_relabel = dict(zip(vids.tolist(), [i for i in range(vids.size(0))]))
        mapping_subg = [vid_relabel[i] for i in o2v_mapping]
        mapping_subg = torch.tensor(mapping_subg, dtype=torch.long) + nids.size(0)  # (batch_size, )

        # mapping from batch to global for v_nodes
        mapping_sub2glb = torch.tensor([self.v_gmap[i] for i in vids.tolist()], dtype=torch.long)
        mappings = {"n2v": mapping_subg, "vs2vg": mapping_sub2glb}

        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index, y=sampled_y)
        return sampled_data, mappings

    def __pre_process(self, data, num_parts, *args):
        """
        need cpu
        edge_index should begin with 0
        """
        x, y, edge_index = data.x, data.y, data.edge_index
        print(f'\033[1;31m Preprocessing data, clustering... \033[0m')
        time_start = time.time()
        x, edge_index = x.to("cpu"), edge_index.to("cpu")
        if y is not None:
            y = y.to("cpu")
        data_pyg = Data(x=x, edge_index=edge_index)
        cluster_data = MyClusterData(data=data_pyg, num_parts=num_parts)
        clusters: list = cluster_data.clusters

        # augment graph
        x_aug = torch.cat([x, torch.zeros(num_parts, x.size(1))], dim=0)  # padding
        y_aug = None
        if y is not None:
            if len(y.size()) > 1:
                y_aug = torch.cat([y, torch.zeros(num_parts, y.size(1))], dim=0)
            else:
                y_aug = torch.cat([y, torch.zeros(num_parts, )])
        idx_base = x.size(0)
        edge_index_aug = edge_index.clone()

        n2v_map = [[], []]  # {nid+vid: vid}
        for i, cluster in enumerate(clusters):
            vnode_idx = idx_base + i
            aug_edges = torch.stack([torch.ones_like(cluster) * vnode_idx, cluster])
            edge_index_aug = torch.cat([edge_index_aug, aug_edges], dim=1)
            aug_edges[[0, 1]] = aug_edges[[1, 0]]
            edge_index_aug = torch.cat([edge_index_aug, aug_edges], dim=1)

            n2v_map[0] += cluster.tolist() + [vnode_idx]
            n2v_map[1] += [vnode_idx] * (cluster.size(0) + 1)

        # output
        data = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug)
        data.n_id = torch.arange(x_aug.size(0))

        vid_relabel = dict([(item, i) for i, item in enumerate(data.n_id[-num_parts:].tolist())])
        n2v_map = dict(zip(*n2v_map))  # {nid+vid: vid}
        # initialize v_nodes' feats: using mean
        v_node_feats = []
        for node_list in clusters:
            v_node_feats.append(torch.mean(x[node_list], dim=0, keepdim=True))
        v_node_feats = torch.cat(v_node_feats, dim=0)

        print(f'\033[1;31m Finish preprocessing data! Use: {time.time() - time_start}s \033[0m')
        return data, data.n_id[:-num_parts], n2v_map, v_node_feats, vid_relabel


class MyClusterData(ClusterData):
    def __init__(self, data: Data, num_parts: int, recursive: bool = False, log: bool = False,
                 save_dir=None):
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        super().__init__(data, num_parts, recursive, log=log, save_dir=save_dir)
        self.clusters = self.get_clusters()

    def get_clusters(self) -> list:
        # TODO test real num < num_parts
        # TODO: what if there is a fake vnode
        #       A: no edges

        adj, partptr, perm = self.data.adj, self.partptr, self.perm

        num_fake_node = 0
        node_idxes = []
        for v_node in range(len(partptr) - 1):
            start, end = partptr[v_node], partptr[v_node + 1]

            # check fake v_node
            if start == end:
                num_fake_node += len(partptr) - 1 - v_node
                break

            node_idx = perm[start:end]
            node_idxes.append(node_idx)

        if num_fake_node > 0:
            raise NotImplemented("num of nodes of vgraph < num_parts")

        return node_idxes


class MyDataLoaderFC:
    def __init__(self, data: Data, num_parts, batch_size: int = -1, use_edge_loss=True, loader=None, shuffle=False):
        if loader is not None:
            raise NotImplemented("Customized loader isn't implemented")

        self.data_aug, self.n_ids, self.vnode_init, self.v_gmap = self.__pre_process(data, num_parts=num_parts)
        self.v_ids = torch.tensor(sorted(list(set(self.v_gmap.keys()))), dtype=torch.long)
        self.num_parts = num_parts

        self.num_n = len(self.n_ids)
        self.num_n_aug = self.data_aug.x.size(0)
        self.batch_size = self.num_n if batch_size == -1 else batch_size
        self.loader = loader
        self.loader_len = math.ceil(self.num_n / self.batch_size)

        self.shuffle = shuffle
        self.batch_nid = self.n_ids
        if self.shuffle:
            self.batch_nid = self.batch_nid[torch.randperm(self.num_n)]

    def __len__(self):
        return self.loader_len

    def __getitem__(self, idx):
        # original nodes
        if idx >= len(self):
            raise StopIteration

        s_o, e_o = self.batch_size * idx, self.batch_size * (idx + 1)
        nids = self.batch_nid[s_o:e_o]  # pay attention to the last batch

        # add vnodes
        idx_aug = torch.cat([nids, self.v_ids], dim=0)
        sampled_x = self.data_aug.x[idx_aug]
        sampled_y = self.data_aug.y[nids] if self.data_aug.y is not None else None  # only need original label
        sampled_edge_index, _ = subgraph(idx_aug, self.data_aug.edge_index, num_nodes=self.num_n_aug,
                                         relabel_nodes=True)

        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index, y=sampled_y)
        return sampled_data

    def __pre_process(self, data, num_parts, *args):
        """
        need cpu
        edge_index should begin with 0
        """
        x, y, edge_index = data.x, data.y, data.edge_index
        print(f'\033[1;31m Preprocessing data, clustering... \033[0m')
        time_start = time.time()
        x, edge_index = x.to("cpu"), edge_index.to("cpu")
        if y is not None:
            y = y.to("cpu")
        data_pyg = Data(x=x, edge_index=edge_index)
        cluster_data = MyClusterData(data=data_pyg, num_parts=num_parts)
        clusters: list = cluster_data.clusters

        # augment x and y
        x_aug = torch.cat([x, torch.zeros(num_parts, x.size(1))], dim=0)  # padding
        y_aug = None
        if y is not None:
            if len(y.size()) > 1:
                y_aug = torch.cat([y, torch.zeros(num_parts, y.size(1))], dim=0)
            else:
                y_aug = torch.cat([y, torch.zeros(num_parts, )])

        # aug edge_index
        idx_base = x.size(0)
        edge_index_aug = edge_index.clone()
        row = torch.arange(idx_base).reshape(-1, 1).repeat(1, len(clusters))
        col = torch.arange(idx_base, idx_base + len(clusters)).reshape(1, -1).repeat(idx_base, 1)
        edge_index_aug_ = torch.stack([row, col]).reshape(2, -1)
        edge_index_aug = torch.cat([edge_index_aug, edge_index_aug_], dim=1)
        edge_index_aug_[[0, 1]] = edge_index_aug_[[1, 0]]
        edge_index_aug = torch.cat([edge_index_aug, edge_index_aug_], dim=1)

        # output
        data = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug)
        data.n_id = torch.arange(x_aug.size(0))

        vid_relabel = dict([(item, i) for i, item in enumerate(data.n_id[-num_parts:].tolist())])
        # initialize v_nodes' feats: using mean
        v_node_feats = []
        for node_list in clusters:
            v_node_feats.append(torch.mean(x[node_list], dim=0, keepdim=True))
        v_node_feats = torch.cat(v_node_feats, dim=0)

        print(f'\033[1;31m Finish preprocessing data! Use: {time.time() - time_start}s \033[0m')
        return data, data.n_id[:-num_parts], v_node_feats, vid_relabel
