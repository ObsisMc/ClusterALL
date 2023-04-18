import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ClusterData
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, degree

import math
import os
import time


class AttnLayer(nn.Module):
    def __init__(self, in_channels, attn_channels, num_parts, heads=1, negative_slope: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.attn_channels = attn_channels
        self.heads = heads
        self.negative_slop = negative_slope
        self.num_parts = num_parts

        self.Ws = nn.Linear(in_channels, heads * attn_channels)
        self.Wd = nn.Linear(in_channels, heads * attn_channels)
        self.bias = nn.Parameter(torch.empty(num_parts, attn_channels))
        nn.init.kaiming_normal_(self.bias)

        self.reset_parameters()

    def forward(self, x_src, x_dst):
        H, C = self.heads, self.attn_channels
        N_src, N_dst = x_src.size(0), x_dst.size(0)
        assert N_dst == self.num_parts

        x_src = self.Ws(x_src).view(H, N_src, C)
        x_dst = self.Wd(x_dst).view(H, N_dst, C)
        x_dst += self.bias.view(1, -1, C)

        alpha = x_src @ x_dst.transpose(-2, -1)  # (H, N_src, N_dst)

        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slop).mean(0)  # (N_src, N_dst)
        alpha = F.softmax(alpha, dim=1)  # (N_src, N_dst)
        return alpha

    def reset_parameters(self):
        self.Ws.reset_parameters()
        self.Wd.reset_parameters()


class AbstractClusteror(nn.Module):
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, decode_channels=None,
                 attn_channels=32, attn_heads=1, dropout=0.1, num_parts=10, **kwargs):
        super().__init__()
        # config
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.decode_channels = decode_channels if decode_channels is not None else hidden_channels
        self.out_channels = out_channels
        self.attn_channels = attn_channels
        self.dropout = dropout
        self.num_parts = num_parts

        # layers
        self.encoder = encoder

        self.fcs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        self.fcs["in2hid"] = nn.Linear(in_channels, hidden_channels)  # encode
        self.fcs["aggr"] = nn.Linear(decode_channels * 2, decode_channels)  # aggregate v_nodes' feats
        self.fcs["output"] = nn.Linear(decode_channels, out_channels)  # decode
        self.bns["ln_dec"] = nn.LayerNorm(decode_channels)
        self.bns["ln_hid"] = nn.LayerNorm(hidden_channels)
        self.activations["elu"] = nn.ELU()

        # cluster: GAT
        self.cluster_attn_layer = AttnLayer(in_channels=decode_channels, attn_channels=attn_channels,
                                            num_parts=num_parts, heads=attn_heads)

        # v_nodes' learnable parameters
        self.vnode_embed = nn.Parameter(torch.randn(self.num_parts, in_channels))  # virtual node
        self.vnode_bias_hid = nn.Parameter(torch.empty(self.num_parts, hidden_channels))
        self.vnode_bias_dcd = nn.Parameter(torch.empty(self.num_parts, decode_channels))
        nn.init.normal_(self.vnode_bias_hid, mean=0, std=0.1)
        nn.init.normal_(self.vnode_bias_dcd, mean=0, std=0.1)

    def reset_parameters(self, init_feat: torch.Tensor):
        self.encoder.reset_parameters()
        for key, item in self.fcs.items():
            self.fcs[key].reset_parameters()
        for key, item in self.bns.items():
            self.bns[key].reset_parameters()
        self.cluster_attn_layer.reset_parameters()
        self.vnode_embed = nn.Parameter(init_feat)

    def forward(self, x, edge_index, mapping=None, **kwargs):
        N = x.size(0) - self.num_parts

        # init v_nodes
        x[-self.num_parts:] = self.vnode_embed
        x = self.activations["elu"](self.bns["ln_hid"](self.fcs["in2hid"](x)))
        x[-self.num_parts:] += self.vnode_bias_hid
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # encode
        x, custom_dict = self.encode_forward(x=x, edge_index=edge_index, **kwargs)
        x = self.activations["elu"](self.bns["ln_encode"](x))

        # cluster
        weight = self.cluster_attn_layer(x[:-self.num_parts], x[-self.num_parts:])
        cluster_idx = torch.argmax(weight, dim=1)
        cluster_idx = torch.cat([cluster_idx, torch.arange(self.num_parts).to(x.device)])
        cluster_idx_ = cluster_idx + N

        # aggr
        x[-self.num_parts:] += self.vnode_bias_dcd
        x = torch.cat([x, x[cluster_idx_]], dim=1)
        x = self.activations["elu"](self.bns["ln_encode"](self.fcs["aggr"](x)))

        # decode
        x = self.fcs["output"](x)
        out = x[:-self.num_parts]

        # interpretability
        cluster_reps = x[-self.num_parts:]
        cluster_mapping = cluster_idx[:-self.num_parts]

        return out, (cluster_reps, cluster_mapping), custom_dict

    def encode_forward(self, x, edge_index, **kwargs):
        output_dict = dict()
        raise NotImplemented("There is no encoder")
        return x, output_dict


class MyDataLoaderCluster:
    def __init__(self, data: Data, num_parts, batch_size: int = -1, loader=None, shuffle=True, eval=False):
        if loader is not None:
            raise NotImplemented("Customized loader isn't implemented")
        self.vnode_init = self.mapping_g = self.mapping_m = None
        self.data_aug, self.n_ids, self.v_gmap = self.__pre_process(data, eval=eval,
                                                                    num_parts=num_parts)
        self.v_ids = torch.tensor(sorted(list(set(self.v_gmap.keys()))), dtype=torch.long)
        self.num_parts = num_parts
        self.eval = eval

        self.num_n = len(self.n_ids)
        self.E = self.data_aug.edge_index.size(1)
        self.v2n_lb = self.E - self.num_n
        self.n2v_lb = self.v2n_lb - self.num_n
        self.v_loop_lb = self.n2v_lb - self.num_parts

        self.num_n_aug = self.data_aug.x.size(0)
        self.batch_size = self.num_n if batch_size == -1 else batch_size
        self.loader = loader
        self.loader_len = math.ceil(self.num_n / self.batch_size)

        self.shuffle = shuffle
        self.batch_nid = self.n_ids
        if self.shuffle:
            self.batch_nid = self.batch_nid[torch.randperm(self.num_n)]
        self.current_bid = -1

    def __len__(self):
        return self.loader_len

    def __getitem__(self, idx):
        # original nodes
        if idx >= len(self):
            self.current_bid = -1
            raise StopIteration

        self.current_bid = idx  # for method update
        s_o, e_o = self.batch_size * idx, self.batch_size * (idx + 1)
        nids = self.batch_nid[s_o:e_o]  # pay attention to the last batch
        batch_size = nids.size(0)

        # add vnodes
        idx_aug = torch.cat([nids, self.v_ids], dim=0)
        sampled_x = self.data_aug.x[idx_aug]
        sampled_y = self.data_aug.y[nids] if self.data_aug.y is not None else None  # only need original label
        sampled_edge_index, _ = subgraph(idx_aug, self.data_aug.edge_index, num_nodes=self.num_n_aug,
                                         relabel_nodes=True)

        # mapping nodes to vnodes in model
        n2v_mapping = None
        if not self.eval:
            n2v_mapping = torch.tensor([self.mapping_m[i] for i in nids.tolist()], dtype=torch.long)
            n2v_mapping += batch_size

        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index, y=sampled_y)
        return sampled_data, n2v_mapping

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

        # augment graph
        x_aug = torch.cat([x, torch.zeros(num_parts, x.size(1))], dim=0)  # padding
        y_aug = None
        if y is not None:
            if len(y.size()) > 1:
                y_aug = torch.cat([y, torch.zeros(num_parts, y.size(1))], dim=0)
            else:
                y_aug = torch.cat([y, torch.zeros(num_parts, )])
        N = x.size(0)
        edge_index_aug = edge_index.clone()
        self_loop_v = torch.arange(N, N + num_parts).view(1, -1).repeat(2, 1)
        edge_index_aug = torch.cat([edge_index_aug, self_loop_v], dim=1)  # (2, [E:E+num_parts+N])

        if not eval:
            data_pyg = Data(x=x, edge_index=edge_index)
            cluster_data = MyClusterData(data=data_pyg, num_parts=num_parts)
            clusters: list = cluster_data.clusters
            # initialize v_nodes' feats: using mean
            v_node_feats = []
            for node_list in clusters:
                v_node_feats.append(torch.mean(x[node_list], dim=0, keepdim=True))
            v_node_feats = torch.cat(v_node_feats, dim=0)
            self.vnode_init = v_node_feats

            nid_key = []  # nid
            vid_graph_item = []  # vid in dataset
            vid_model_item = []  # vid in model

            sorted_n_edge_index_ = torch.empty((2, N), dtype=torch.long)  # vid -> sorted_nid
            for i, cluster in enumerate(clusters):
                vnode_idx = N + i
                edge_index_ = torch.stack([torch.ones_like(cluster) * vnode_idx, cluster])  # vid -> nid
                sorted_n_edge_index_[:, cluster] = edge_index_

                nid_key += cluster.tolist()
                vid_graph_item += [vnode_idx] * cluster.size(0)
                vid_model_item += [i] * cluster.size(0)
            edge_index_aug = torch.cat([edge_index_aug, sorted_n_edge_index_[[1, 0]]],
                                       dim=1)  # (2, [E+num_parts:E+num_parts+N])
            edge_index_aug = torch.cat([edge_index_aug, sorted_n_edge_index_],
                                       dim=1)  # (2, [E+num_parts+N:E+num_parts+2N])

            n2v_gmap = dict(zip(nid_key, vid_graph_item))  # {nid: vid}, vid begins with len(nid)
            n2v_model = dict(zip(nid_key, vid_model_item))  # {nid: vid}, vid begins with 0
            self.mapping_g = n2v_gmap
            self.mapping_m = n2v_model
        else:
            pass

        # output
        data = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug)
        data.n_id = torch.arange(x_aug.size(0))

        vid_relabel = dict([(item, i) for i, item in enumerate(data.n_id[-num_parts:].tolist())])

        print(f'\033[1;31m Finish preprocessing data! Use: {time.time() - time_start}s \033[0m')
        return data, data.n_id[:-num_parts], vid_relabel

    def update_cluster(self, mapping, log=True):
        if self.eval:
            raise NotImplemented("update_cluster isn't implemented in testing part")

        batch_idx, sorted_v_ids, device = self.current_bid, self.v_ids, self.data_aug.edge_index
        bound1, bound2 = self.n2v_lb, self.v2n_lb
        g_mapping = mapping + self.num_n
        n_ids = self.batch_nid[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

        # log
        if log:
            cmp = torch.stack([self.data_aug.edge_index[1, n_ids + bound1], g_mapping.to(device)])
            change_idx = cmp[0] != cmp[1]
            num_change = torch.sum(change_idx)
            flow = cmp[:, change_idx] - self.num_n
            d_out, d_in = degree(flow[0], self.num_parts), degree(flow[1], self.num_parts)
            change_dict = dict([(i, f) for i, f in enumerate((d_in - d_out).tolist())])
            print(f"{num_change}/{n_ids.size(0)} nodes change clusters: "
                  f"{change_dict if self.num_parts < 10 else 'too long'}")

        self.data_aug.edge_index[1, n_ids + bound1] = g_mapping.to(device)
        self.data_aug.edge_index[0, n_ids + bound2] = g_mapping.to(device)


class MyClusterData(ClusterData):
    def __init__(self, data: Data, num_parts: int, recursive: bool = False, log: bool = False,
                 save_dir=None):
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        super().__init__(data, num_parts, recursive, log=log, save_dir=save_dir)
        self.clusters = self.get_clusters()

    def get_clusters(self) -> list:
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
            raise NotImplemented(
                f"The graph cannot be split to {self.num_parts} clusters, please try smaller num_parts")

        return node_idxes


def main():
    pass


if __name__ == "__main__":
    main()
