import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ClusterData
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, degree
from torch_geometric.transforms import ToSparseTensor
from typing import Any, Union
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
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, decode_channels, num_parts,
                 attn_channels=32, attn_heads=1, dropout=0, aggr_type="weighted", **kwargs):
        super().__init__()
        # config
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.decode_channels = decode_channels if decode_channels is not None else hidden_channels
        self.out_channels = out_channels
        self.attn_channels = attn_channels
        self.dropout = dropout
        self.num_parts = num_parts
        self.aggr_type = aggr_type
        assert aggr_type == "weighted" or aggr_type == "max"

        # layers
        self.encoder = encoder

        self.fcs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        self.fcs["in2hid"] = nn.Linear(in_channels, hidden_channels)  # encode
        self.fcs["aggr"] = nn.Linear(self.decode_channels * 2, self.decode_channels)  # aggregate v_nodes' feats
        self.fcs["output"] = nn.Linear(self.decode_channels, out_channels)  # decode
        self.bns["ln_dec"] = nn.LayerNorm(self.decode_channels)
        self.bns["ln_hid"] = nn.LayerNorm(hidden_channels)
        self.activations["elu"] = nn.ELU()

        # cluster: attention
        self.cluster_attn_layer = AttnLayer(in_channels=self.decode_channels, attn_channels=attn_channels,
                                            num_parts=num_parts, heads=attn_heads)

        # v_nodes' learnable parameters
        self.vnode_embed = nn.Parameter(torch.randn(self.num_parts, in_channels))  # virtual node
        self.vnode_bias_hid = nn.Parameter(torch.empty(self.num_parts, hidden_channels))
        self.vnode_bias_dcd = nn.Parameter(torch.empty(self.num_parts, self.decode_channels))
        nn.init.normal_(self.vnode_bias_hid, mean=0, std=0.1)
        nn.init.normal_(self.vnode_bias_dcd, mean=0, std=0.1)

    def reset_parameters(self, init_feat=None):
        assert self.num_parts > 0 and init_feat is not None or self.num_parts == 0 and init_feat is None
        self.encoder.reset_parameters()
        for key, item in self.fcs.items():
            self.fcs[key].reset_parameters()
        for key, item in self.bns.items():
            self.bns[key].reset_parameters()
        self.cluster_attn_layer.reset_parameters()
        if self.num_parts > 0:
            self.vnode_embed = nn.Parameter(init_feat)

    def encode_forward(self, x, edge_index, **kwargs) -> (torch.Tensor, dict):
        output_dict = dict()
        raise NotImplemented("There is no encoder")
        return x, output_dict

    def forward(self, x, edge_index, **kwargs):
        device = x.device
        N = x.size(0) - self.num_parts
        node_mask = torch.zeros((x.size(0),), dtype=torch.bool).to(device)
        node_mask[:N] = True  # True for nodes, False for v_nodes

        # init v_nodes
        if self.num_parts > 0:
            x[~node_mask] = self.vnode_embed
        x = self.activations["elu"](self.bns["ln_hid"](self.fcs["in2hid"](x)))
        if self.num_parts > 0:
            x[~node_mask] += self.vnode_bias_hid
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encode
        x, custom_dict = self.encode_forward(x=x, edge_index=edge_index, **kwargs)
        x = self.activations["elu"](self.bns["ln_dec"](x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_parts > 0:
            # cluster
            weight = self.cluster_attn_layer(x[node_mask], x[~node_mask])  # (N, num_parts)
            weight = torch.cat([weight, torch.eye(self.num_parts).to(device)])
            # get the next cluster
            cluster_idx = torch.argmax(weight[node_mask], dim=1)  # (N,)
            # aggr
            x[~node_mask] += self.vnode_bias_dcd
            weighted_clusters = weight @ x[~node_mask]
            x = torch.cat([x, weighted_clusters], dim=1)
            x = self.activations["elu"](self.bns["ln_dec"](self.fcs["aggr"](x)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # decode
        x = self.fcs["output"](x)
        out = x[node_mask]

        # interpretability
        cluster_reps = x[~node_mask]
        cluster_mapping = cluster_idx if self.num_parts > 0 else torch.empty((0,)).to(device)

        return out, (cluster_reps, cluster_mapping), custom_dict


class AbstractClusterDataset:
    def __init__(self, dataset: Any, data: Data, split_idx: dict, num_parts: int, load_path: str, transform=None):
        self.__init_dataset(dataset)
        self.num_parts__ = num_parts

        data = self.__process_data(data, transform)
        self.transform = transform
        self.N__, self.E__ = data.x.size(0), data.edge_index.size(1)
        self.train_idx__, self.valid_idx__, self.test_idx__ = split_idx["train"], split_idx["valid"], split_idx["test"]
        assert torch.all(self.train_idx__[1:] - self.train_idx__[:-1] > 0)
        assert torch.all(self.valid_idx__[1:] - self.valid_idx__[:-1] > 0)
        assert torch.all(self.test_idx__[1:] - self.test_idx__[:-1] > 0)

        # cluster_lst: [vid0, vid1, ..., vidn] relabeled vid,only training nodes store clusters' info
        self.data_aug__, self.n_aug_ids__, self.vnode_init__, self.cluster_lst__ = self.__pre_process(data,
                                                                                                      num_parts,
                                                                                                      load_path)  # data is the whole graph
        self.N_aug__, self.E_aug__ = self.data_aug__.x.size(0), self.data_aug__.edge_index.size(1)
        self.v_ids__ = self.n_aug_ids__[-num_parts:] if num_parts > 0 else torch.empty((0,), dtype=torch.long)
        self.N_train__ = len(self.train_idx__)

        # used by get_split_data,
        # v_nodes must align with training nodes, for simplicity, put training nodes in head of idx
        self.train_idx_aug__ = torch.cat([self.train_idx__])
        self.valid_idx_aug__ = torch.cat([self.train_idx__, self.valid_idx__])
        self.test_idx_aug__ = torch.cat([self.train_idx__, self.test_idx__])
        self.all_idx_aug__ = torch.cat([self.train_idx__, self.valid_idx__, self.test_idx__])

    def __process_data(self, data, transform):
        """
        If has adj_t, convert it to edge_index
        """
        device = data.x.device
        if isinstance(transform, ToSparseTensor):
            row, col, _ = data.adj_t.t().coo()
            data = Data(x=data.x, y=data.y, edge_index=torch.stack([row, col]).to(device))
        return data

    def __init_dataset(self, dataset):
        for key in dir(dataset):
            if key.startswith("__") and key.endswith("__"):
                continue
            if not callable(dataset.__getattribute__(key)):
                self.__setattr__(key, dataset.__getattribute__(key))

    def get_init_vnode(self, device):
        return self.vnode_init__.to(device)

    def get_split_data(self, split_name):
        """
        Any split_names that aren't 'train' are for inference
        """
        x_aug, y_aug, edge_index_aug = self.data_aug__.x, self.data_aug__.y, self.data_aug__.edge_index
        if split_name == "train":
            idx = self.train_idx_aug__
        elif split_name == "valid":
            idx = self.valid_idx_aug__
        elif split_name == "test":
            idx = self.test_idx_aug__
        elif split_name == "all":
            idx = self.all_idx_aug__
        else:
            raise ValueError("No such split_name, choose from ('train','valid','test','all')")
        idx_aug = torch.cat([idx, self.v_ids__])
        x_aug, y_aug = x_aug[idx_aug], y_aug[idx]  # y_aug doesn't contain v_nodes
        edge_index_aug, _ = subgraph(idx_aug, edge_index_aug, num_nodes=self.N_aug__, relabel_nodes=True)

        # IMPORT: need to perm edge_index
        # since x is permuted and edge_index may not match x (subgraph doesn't perm edge_index)
        # support train/valid/test set all have sorted idx
        idx_perm = torch.argsort(idx_aug)
        edge_index_aug_perm = idx_perm[edge_index_aug]

        data_split = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug_perm)
        return data_split, idx

    def __pre_process(self, data, num_parts, load_path=None):
        """
        need cpu
        edge_index should begin with 0
        """
        if num_parts == 0:
            return data, torch.arange(data.x.size(0)), torch.empty((0, data.x.size(1))), None
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

        # get train set, relabel idx
        train_x = x[self.train_idx__]
        train_edge_index, _ = subgraph(self.train_idx__, edge_index, num_nodes=N, relabel_nodes=True)  # relabel idx
        N_train = train_x.size(0)

        if load_path is None:
            train_data = Data(x=train_x, edge_index=train_edge_index)
            clustered_data = MyClusterData(data=train_data, num_parts=num_parts)
            clusters: list = clustered_data.clusters  # use relabel idx
            # initialize v_nodes' feats: using mean
            v_node_feats = []
            for node_list in clusters:
                v_node_feats.append(torch.mean(train_x[node_list], dim=0, keepdim=True))
            v_node_feats = torch.cat(v_node_feats, dim=0)

            nid_key = []  # relabeled nid
            vid_model_item = []  # relabeled vid in model

            sorted_n_edge_index_ = torch.empty((2, N_train), dtype=torch.long)  # vid -> sorted_nid
            for i, cluster in enumerate(clusters):
                vnode_idx = N + i
                edge_index_ = torch.stack(
                    [torch.ones_like(cluster) * vnode_idx, self.train_idx__[cluster]])  # vid -> nid
                sorted_n_edge_index_[:, cluster] = edge_index_

                nid_key += cluster.tolist()
                vid_model_item += [i] * cluster.size(0)
            edge_index_aug = torch.cat([edge_index_aug, sorted_n_edge_index_[[1, 0]]],
                                       dim=1)  # (2, [E+num_parts:E+num_parts+N_train])
            edge_index_aug = torch.cat([edge_index_aug, sorted_n_edge_index_],
                                       dim=1)  # (2, [E+num_parts+N_train:E+num_parts+2N_train])

            cluster_mapping = list(zip(nid_key, vid_model_item))  # {relabeled nid: relabeled vid}, vid begins with 0
            cluster_idx_lst = sorted(cluster_mapping, key=lambda e: e[0])
            cluster_idx_lst = torch.tensor(cluster_idx_lst)[:, 1].view(-1, )
        else:
            print(f'\033[1;31m Loading cluster from {load_path} \033[0m')
            v_node_feats = None
            edge_index_cluster = self.load_cluster(load_path).to(edge_index_aug.device)
            edge_index_aug = torch.cat([edge_index_aug, edge_index_cluster], dim=1)

            cluster_idx_lst = edge_index_cluster[0, -self.N_train__:] - self.N__

        # output
        data = Data(x=x_aug, y=y_aug, edge_index=edge_index_aug)
        data.n_id = torch.arange(x_aug.size(0))

        print(f'\033[1;31m Finish preprocessing data! Use: {time.time() - time_start}s \033[0m')
        return data, data.n_id, v_node_feats, cluster_idx_lst

    def save_cluster(self, save_dir, save_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cluster_n = self.num_parts__
        edge_index_cluster = self.data_aug__.edge_index[:, -self.N_train__ * 2:]

        info = {"cluster_n": cluster_n, "edge_index_cluster": edge_index_cluster}
        torch.save(info, os.path.join(save_dir, save_name))

    def load_cluster(self, save_path):
        assert os.path.exists(save_path)
        info = torch.load(save_path)
        assert info["cluster_n"] == self.num_parts__
        return info["edge_index_cluster"]


class AbstractClusterLoader:
    def __init__(self, dataset: AbstractClusterDataset, split_name: str, is_eval: bool, batch_size: int, shuffle):
        """
        when split_name isn't "train", it is in eval mode, then batch size must be full batch and shuffle is False
        """
        self.split_name = split_name
        if split_name != "train":
            self.is_eval = True
        else:
            self.is_eval = is_eval

        self.dataset = dataset
        self.transform = dataset.transform
        self.data_aug, self.idx_cvt = dataset.get_split_data(split_name)
        self.num_parts = dataset.num_parts__
        # self.v_ids = torch.tensor(sorted(list(set(self.v_gmap.keys()))), dtype=torch.long)

        # nodes' num
        self.N_aug, self.N = self.data_aug.x.size(0), self.data_aug.x.size(0) - self.num_parts
        self.v_ids = torch.arange(self.N, self.N_aug)
        # edges' num
        self.E_vedge = self.N_train = dataset.N_train__ if self.num_parts > 0 else 0
        self.E_aug = self.data_aug.edge_index.size(1)
        self.E = self.E_aug - self.E_vedge * 2 - self.num_parts  # [E:E+num_parts] for self loop of v_nodes

        # mapping, mapping_model + N = mapping_graph
        self.mapping_model = dataset.cluster_lst__  # (N_train,)
        self.mapping_graph = self.mapping_model + self.N if self.mapping_model is not None else None  # (N_train,)
        # batch
        self.batch_size = self.N if batch_size == -1 or self.is_eval else batch_size
        self.batch_num = math.ceil(self.N / self.batch_size)

        self.shuffle = shuffle if not self.is_eval else False
        self.batch_nid = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        self.current_bid = -1

    def convert(self, x):
        device = x.device
        if self.split_name == "train":
            return x
        elif self.split_name == "all":
            out_x = torch.empty_like(x).to(device)
            out_x[self.idx_cvt] = x
        else:
            out_x = x[self.N_train:]

        return out_x

    def eval(self):
        self.update_batch_size(-1, False)
        return self

    def update_batch_size(self, batch_size, shuffle=False):
        self.batch_size = self.N if batch_size == -1 or self.is_eval else batch_size
        self.batch_num = math.ceil(self.N / self.batch_size)

        self.shuffle = shuffle if not self.is_eval else False
        self.batch_nid = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        if idx >= len(self):
            self.current_bid = -1
            raise StopIteration

        self.current_bid = idx  # for method update
        s_o, e_o = self.batch_size * idx, self.batch_size * (idx + 1)
        n_ids = self.batch_nid[s_o:e_o]  # pay attention to the last batch
        batch_size = n_ids.size(0)  # the last batch <= self.batch_size

        # add v_nodes
        idx_aug = torch.cat([n_ids, self.v_ids], dim=0)
        sampled_x = self.data_aug.x[idx_aug]
        sampled_y = self.data_aug.y[n_ids] if self.data_aug.y is not None else None  # only need original label
        sampled_edge_index, _ = subgraph(idx_aug, self.data_aug.edge_index, num_nodes=self.N_aug,
                                         relabel_nodes=True)
        # IMPORT: need to perm edge_index
        idx_perm = torch.argsort(idx_aug)
        sampled_edge_index_perm = idx_perm[sampled_edge_index]

        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index_perm, y=sampled_y)
        if self.transform is not None:
            sampled_data = self.transform(sampled_data)
        return sampled_data

    def update_cluster(self, mapping, log=True):
        """
        mapping (batch_size,)
        """
        if self.is_eval:
            raise NotImplemented("Only training set can update cluster")
        if self.num_parts == 0:
            return

        batch_idx, device = self.current_bid, self.data_aug.edge_index.device
        s_o, e_o = self.batch_size * batch_idx, self.batch_size * (batch_idx + 1)
        n_ids = self.batch_nid[s_o:e_o]

        mapping_split = mapping + self.N
        bound1, bound2 = self.E + self.num_parts, self.E_aug - self.E_vedge
        # log
        if log:
            cmp = torch.stack([self.data_aug.edge_index[1, n_ids + bound1], mapping_split.to(device)])
            change_idx = cmp[0] != cmp[1]
            num_change = torch.sum(change_idx)
            flow = cmp[:, change_idx] - self.N
            d_out, d_in = degree(flow[0], self.num_parts), degree(flow[1], self.num_parts)
            change_dict = dict([(i, f) for i, f in enumerate((d_in - d_out).tolist())])
            print(f"{num_change}/{n_ids.size(0)} nodes change clusters: "
                  f"{change_dict if self.num_parts < 10 else 'too long'}")

        self.data_aug.edge_index[1, n_ids + bound1] = mapping_split.to(device)
        self.data_aug.edge_index[0, n_ids + bound2] = mapping_split.to(device)

        # update dataset's edge_index
        mapping_global = mapping + self.dataset.N__
        bound1_global, bound2_global = self.dataset.E__ + self.num_parts, self.dataset.E_aug__ - self.E_vedge
        self.dataset.data_aug__.edge_index[1, n_ids + bound1_global] = mapping_global.to(device)
        self.dataset.data_aug__.edge_index[0, n_ids + bound2_global] = mapping_global.to(device)


class MyClusterData(ClusterData):
    def __init__(self, data: Data, num_parts: int):
        super().__init__(data, num_parts, False, log=False, save_dir=None)
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
            raise NotImplemented(f"The graph cannot be split to {self.num_parts} clusters, please try a smaller value")

        return node_idxes
