import torch

from typing import Optional, Union
import copy
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_sparse import SparseTensor, cat
import os


class MyClusterData(ClusterData):
    def __init__(self, data: Data, num_parts: int, v_graph=True, recursive: bool = False, log: bool = False,
                 save_dir=None):
        data["edge_attr"] = data.edge_attr if data.edge_attr is not None else torch.ones((data.edge_index.size(1), 1))
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        super().__init__(data, num_parts, recursive, log=log, save_dir=save_dir)
        self.device = data.x.device
        self.o_edge_index = data.edge_index
        self.o_edge_attr = data.edge_attr

        self.v_graph_edge_index = None
        self.v_graph_edge_attr = None
        self.v_graph_num_nodes = None
        self.v_graph_num_edges = None
        self.v_graph_partptr = None

        if v_graph:
            self._find_v_graph()

    def _find_v_graph(self):
        # TODO consider 1 cluster, test real num < num_parts
        # TODO: what if there is a fake vnode
        #       A: no edges
        # TODO: what if there is a vnode with no edge:
        #       A: ...

        adj, partptr, perm = self.data.adj, self.partptr, self.perm

        v_graph_edge_index = [[], []]
        v_graph_edge_attr = None
        v_graph_num_edges = 0
        num_fake_node = 0
        v_graph_partptr = [0]
        for v_node1 in range(len(partptr) - 2):
            n1_s, n1_e = partptr[v_node1], partptr[v_node1 + 1]
            last_turn = v_node1 + 3 == len(partptr)
            if n1_s == n1_e:  # if vnode1 has no nodes, it is a fake vnode
                num_fake_node += 1
                if not last_turn:  # when it is the final loop, need to consider whether vnode2 is fake
                    continue
            if last_turn:
                if partptr[v_node1 + 1] == partptr[v_node1 + 2]:
                    num_fake_node += 1
                    break

            for v_node2 in range(v_node1 + 1, len(partptr) - 1):
                n2_s, n2_e = partptr[v_node2], partptr[v_node2 + 1]
                if n2_s == n2_e:  # if vnode2 has no nodes, it is a fake vnode
                    continue
                _, _, original_edge_idx = adj[n1_s:n1_e, n2_s:n2_e].coo()
                _, _, original_edge_idx_r = adj[n2_s:n2_e, n1_s:n1_e].coo()

                # there may be no edge between vnode1 and vnode2
                v_graph_num_edges += len(original_edge_idx) + len(original_edge_idx_r)
                # if partptr[i] == partptr[i+1], there must be two vnodes that have no inter-edge with each other
                v_graph_partptr.append(v_graph_num_edges)

                v_graph_edge_index[0] += [v_node1 for _ in range(len(original_edge_idx))]
                v_graph_edge_index[0] += [v_node2 for _ in range(len(original_edge_idx_r))]
                v_graph_edge_index[1] += [v_node2 for _ in range(len(original_edge_idx_r))]
                v_graph_edge_index[1] += [v_node1 for _ in range(len(original_edge_idx))]

                original_edge_idx_all = torch.cat([original_edge_idx, original_edge_idx_r], dim=0).long()
                # pay attention: a cluster may be a fake one (has no node in it)
                if v_graph_edge_attr is None:
                    v_graph_edge_attr = self.o_edge_attr[original_edge_idx_all, :]
                else:
                    v_graph_edge_attr = torch.cat(
                        [v_graph_edge_attr, self.o_edge_attr[original_edge_idx_all, :]], dim=0)

        self.v_graph_edge_index = torch.Tensor(v_graph_edge_index).to(self.device)
        self.v_graph_edge_attr = v_graph_edge_attr.to(self.device)
        self.v_graph_num_nodes = len(partptr) - 1 - num_fake_node  # real num of clusters
        self.v_graph_num_edges = v_graph_num_edges
        self.v_graph_partptr = torch.Tensor(v_graph_partptr).long().to(self.device)

        if self.v_graph_num_nodes < len(partptr) - 1:
            raise NotImplemented("num of nodes of vgraph < num_parts")

    @property
    def virtual_graph(self, aggregate=True):
        if self.v_graph_edge_attr is None or self.v_graph_edge_index is None:
            self._find_v_graph()
        if aggregate:
            return self.aggregate_v_edge(self.v_graph_edge_index, self.v_graph_edge_attr, self.v_graph_partptr)
        return self.v_graph_edge_index, self.v_graph_edge_attr, self.v_graph_partptr

    def aggregate_v_edge(self, o_edge_index: torch.Tensor, o_edge_attr: torch.Tensor, partptr: torch.Tensor):
        v_edge_index = [[], []]
        v_edge_attr = []
        device = o_edge_attr.device
        # TODO consider num of vnodes < num_parts, if so, index of vnodes need shift
        for i in range(1, partptr.size(0)):
            s, e = partptr[i - 1], partptr[i]
            if s == e:  # no inter-edge between a certain pair of vnodes
                continue
            vn1, vn2 = o_edge_index[0, s].item(), o_edge_index[1, s].item()
            v_edge_index[0] += [vn1, vn2]
            v_edge_index[1] += [vn2, vn1]

            v_attrs = o_edge_attr[s:e, :]  # (N, C)
            v_attrs = torch.mean(v_attrs, dim=0, keepdim=True)
            v_edge_attr += [v_attrs, v_attrs]

        v_edge_index = torch.Tensor(v_edge_index).long().to(device)
        v_edge_attr = torch.cat(v_edge_attr, dim=0).to(device)
        return v_edge_index, v_edge_attr
