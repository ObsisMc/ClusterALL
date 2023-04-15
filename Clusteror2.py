import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, softmax, remove_self_loops, add_self_loops
from torch_geometric.nn import GATConv, GAT

import tqdm
import math
import os
from typing import Union, Tuple


class AttnLayer(GATConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # We first transform the input node features.
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            num_nodes = min(size) if size is not None else num_nodes
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        return edge_index, alpha


class Clusteror(nn.Module):
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, num_layers, attn_channels=64,
                 use_jk=False, dropout=0.1, num_parts=10):
        super().__init__()

        self.dropout = dropout
        self.num_parts = num_parts
        decode_channels = hidden_channels * num_layers + hidden_channels if use_jk else hidden_channels
        self.encoder = encoder
        self.gat = GATConv(hidden_channels, decode_channels)
        self.test_encoder = GATConv(in_channels, decode_channels)

        self.fcs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        # encode
        self.fcs["proj_in"] = nn.Linear(in_channels, in_channels)
        self.bns["ln_in"] = nn.LayerNorm(in_channels)
        self.bns["ln_encode"] = nn.LayerNorm(decode_channels)
        self.activations["elu"] = nn.ELU()

        self.fcs["in2hid"] = nn.Linear(in_channels, hidden_channels)
        self.bns["ln_hid"] = nn.LayerNorm(hidden_channels)

        # cluster: GAT
        # self.cluster_attn_layer = AttnLayer(decode_channels, attn_channels, heads=2, add_self_loops=False)
        # self.node_attn_layer = GATConv(decode_channels, decode_channels)

        # aggregate vnodes' feat
        self.fcs["aggr"] = nn.Linear(decode_channels * 2, decode_channels)

        # decode
        self.fcs["output"] = nn.Linear(decode_channels, out_channels)

        # vnodes
        self.vnode_embed = nn.Parameter(torch.randn(self.num_parts, in_channels))  # virtual node
        self.vnode_bias_hid = nn.Parameter(torch.empty(self.num_parts, hidden_channels))
        self.vnode_bias_dcd = nn.Parameter(torch.empty(self.num_parts, decode_channels))
        nn.init.normal_(self.vnode_bias_hid, mean=0, std=0.1)
        nn.init.normal_(self.vnode_bias_dcd, mean=0, std=0.1)

    def forward(self, x, **kwargs):
        # set clusters explicit
        # out, loss = self.foward_cluster(x, **kwargs)

        # end to end: get embedding and find clusters
        out, link_loss, infos = self.forward_fc(x, **kwargs)
        return out, link_loss, infos

    def forward_fc(self, x, **kwargs):
        adjs, tau = kwargs["adjs"], kwargs.get("tau", 0.25)
        mapping = kwargs.get("mapping")
        edge_index = adjs[0]
        n = x.size(0) - self.num_parts

        # init v_nodes
        x[-self.num_parts:] = self.vnode_embed
        x = self.activations["elu"](self.bns["ln_hid"](self.fcs["in2hid"](x)))
        x[-self.num_parts:] += self.vnode_bias_hid
        # encode
        x, loss, weight = self.encoder(x, adjs=adjs, tau=tau, num_parts=self.num_parts)
        # x = self.gat(x, edge_index)
        # x = self.test_encoder(x, edge_index)
        # loss, weight = [0], None
        # x[-self.num_parts:] += torch.randn_like(x[-self.num_parts:])

        # analysis
        density = (edge_index.size(1) - self.num_parts * n * 2) / (n * math.log(n))
        v_edge_density = self.num_parts * n * 2 / (edge_index.size(1) - self.num_parts * n * 2)
        v_smoothing_std = torch.mean(torch.std(x[-self.num_parts:], dim=0))

        # cluster
        if mapping is not None:
            cluster_id = mapping.to(x.device)
        else:
            weight_v = weight[-1][1]  # (N,K)
            # n = x.size(0) - self.num_parts
            # v2n_w = weight_final[-n * self.num_parts:].reshape(n, -1)  # (N,K)
            cluster_id = torch.argmax(weight_v, dim=1, keepdim=False)
        cluster_idx = cluster_id + n
        # edge_clusters = edge_index[:, -self.num_parts * n:]
        # edge_clusters_, cluster_attn = self.cluster_attn_layer(x, edge_clusters)
        # cluster_attn = cluster_attn.mean(dim=-1)
        # cluster_idx = torch.argmax(cluster_attn.reshape(n, -1), dim=-1, keepdim=True)
        # cluster_idx += n

        # aggr
        x[-self.num_parts:] += self.vnode_bias_dcd
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
        cluster_mapping = cluster_id

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
    def __init__(self, data: Data, num_parts, batch_size: int = -1, eval=False, warmup_epoch=0, shuffle=False):
        self.eval = eval
        self.init_mapping = self.vnode_init = None
        self.data_aug, self.n_ids, self.v_gmap = self.__pre_process(data, num_parts=num_parts,
                                                                    eval=eval)
        self.v_ids = torch.tensor(sorted(list(set(self.v_gmap.keys()))), dtype=torch.long)
        self.num_parts = num_parts

        self.num_n = len(self.n_ids)
        self.num_n_aug = self.data_aug.x.size(0)
        self.batch_size = self.num_n if batch_size == -1 else batch_size
        self.loader_len = math.ceil(self.num_n / self.batch_size)

        self.shuffle = shuffle
        self.batch_nid = self.n_ids
        if self.shuffle:
            self.batch_nid = self.batch_nid[torch.randperm(self.num_n)]
        self.warmup_epoch = warmup_epoch

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

        # init mapping
        mapping = None
        if not self.eval:
            mapping = torch.tensor([self.init_mapping[i] for i in nids.tolist()], dtype=torch.long)
        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index, y=sampled_y)
        return sampled_data, mapping

    def __pre_process(self, data, num_parts, eval=False, **kwargs):
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
        if not eval:
            data_pyg = Data(x=x, edge_index=edge_index)
            cluster_data = MyClusterData(data=data_pyg, num_parts=num_parts)
            clusters: list = cluster_data.clusters

            # initialize v_nodes' feats: using mean
            v_node_feats = []
            n2v_mapping = [[], []]
            for i, node_list in enumerate(clusters):
                v_node_feats.append(torch.mean(x[node_list], dim=0, keepdim=True))
                n2v_mapping[0] += node_list.tolist()
                n2v_mapping[1] += [i] * len(node_list)
            self.init_mapping = dict(zip(*n2v_mapping))
            self.vnode_init = torch.cat(v_node_feats, dim=0)

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
        row = torch.arange(idx_base).reshape(-1, 1).repeat(1, num_parts)  # n_ids
        col = torch.arange(idx_base, idx_base + num_parts).reshape(1, -1).repeat(idx_base, 1)  # v_ids
        edge_index_aug_ = torch.stack([row, col]).reshape(2, -1)
        edge_index_aug = torch.cat([edge_index_aug, edge_index_aug_], dim=1)
        edge_index_aug_[[0, 1]] = edge_index_aug_[[1, 0]]
        edge_index_aug = torch.cat([edge_index_aug, edge_index_aug_], dim=1)

        # output
        data = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug)
        data.n_id = torch.arange(x_aug.size(0))

        vid_relabel = dict([(item, i) for i, item in enumerate(data.n_id[-num_parts:].tolist())])

        print(f'\033[1;31m Finish preprocessing data! Use: {time.time() - time_start}s \033[0m')
        return data, data.n_id[:-num_parts], vid_relabel


def main():
    import scipy.sparse as sp
    # x = torch.randn((6,5))
    # edge_index = torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3],
    #                            [1, 0, 2, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 5]], dtype=torch.long)

    n = 2000
    adj = torch.ones((n, n))
    x = torch.randn((n, 5))
    coo_m = sp.coo_matrix(adj)
    edge_index = torch.stack([torch.tensor(coo_m.row).long(), torch.tensor(coo_m.col).long()])

    data = Data(x, edge_index)
    num_parts = 10
    loader = MyDataLoaderFC(data, num_parts, -1)
    gat = GATConv(5, 5)
    for sampled_data, mapping in loader:
        sampled_data.x[-num_parts:] = torch.randn((num_parts, 5))
        out = gat(sampled_data.x, sampled_data.edge_index)
        print(out)


if __name__ == "__main__":
    main()
