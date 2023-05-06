import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected
import torch
import torch.nn.functional as F
import colorsys
import random


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


def test_draw_graph():
    N = 3
    x = torch.randn((N, 3))
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2],
                               [1, 0, 2, 0, 2, 1]], dtype=torch.long)

    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)

    G = to_networkx(data)
    fix_pos = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    positions = nx.spring_layout(G, pos=fix_pos)

    colors = list(map(lambda x: color(tuple(x)), ncolors(3)))

    fig, ax = plt.subplots()
    nx.draw(G, pos=positions, node_color=colors, ax=ax, arrows=False, with_labels=True)
    plt.show()


def draw_graph(data: Data, pos: list = None, random_color: bool = False):
    G = to_networkx(data)
    n = G.number_of_nodes()

    fix_pos = dict(zip([i for i in range(n)], pos))
    pos = nx.spring_layout(G, pos=fix_pos)

    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, ax=ax, arrows=False, with_labels=False)
    plt.show()


def draw_graph_clusters(data: Data, pos: torch.Tensor, cluster_mapping: torch.Tensor, with_label: bool = False,
                        arrows=False):
    G = to_networkx(data)
    n = G.number_of_nodes()
    cluster_ids = torch.unique(cluster_mapping, return_counts=False)
    color_map = list(map(lambda x: color(tuple(x)), ncolors(len(cluster_ids))))

    fix_pos = dict(zip([i for i in range(n)], pos.tolist()))
    # pos = nx.spring_layout(G, pos=fix_pos)
    colors = [color_map[i] for i in cluster_mapping.tolist()]
    fig, ax = plt.subplots()
    nx.draw(G, pos=fix_pos, ax=ax, node_color=colors, arrows=arrows, with_labels=with_label)
    plt.show()


def smooth_avg(x, win_size=10):
    size = x.size()
    if len(size) < 2:
        x = x.reshape(1, -1)
    assert len(x.size()) == 2

    padding_x = torch.cat([x[:, 0].reshape(-1, 1).repeat(1, win_size - 1), x], dim=1)
    avg_x = F.avg_pool1d(padding_x, win_size, stride=1)
    if len(size) < 2:
        avg_x = avg_x.squeeze(0)
    return avg_x


if __name__ == "__main__":
    N = 4
    x = torch.randn((N, 3))
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2, 2],
                               [1, 0, 2, 0, 2, 1, 3]], dtype=torch.long)

    # edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    pos = torch.tensor([[1, 1], [2, 2], [3, 5], [3, 3]])
    mapping = torch.tensor([0, 0, 0, 1])
    draw_graph_clusters(data, pos, mapping, arrows=True, with_label=True)
