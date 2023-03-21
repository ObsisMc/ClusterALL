from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph


def get_adjs(e_idx, rb_order_inner, n_inner, device="cpu"):
    adjs_inner = []
    adj_inner, _ = remove_self_loops(e_idx)
    adj_inner, _ = add_self_loops(adj_inner, num_nodes=n_inner)
    adjs_inner.append(adj_inner.to(device))
    # for _ in range(rb_order_inner - 1):  # edge_index of high order adjacency
    #     adj_inner = adj_mul(adj_inner, adj_inner, n_inner)
    #     adjs_inner.append(adj_inner)
    return adjs_inner
