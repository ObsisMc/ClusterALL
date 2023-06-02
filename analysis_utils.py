import torch
import torch.nn.functional as F
import numpy as np
from analysis import Analyzer
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch_geometric.utils import to_networkx
import networkx as nx
import colorsys
import random

rcParams['font.family'] = 'SimHei'


def smooth_avg(x, win_size=10):
    x = torch.tensor(x)
    size = x.size()
    if len(size) < 2:
        x = x.reshape(1, -1)
    assert len(x.size()) == 2

    padding_x = torch.cat([x[:, 0].reshape(-1, 1).repeat(1, win_size - 1), x], dim=1)
    avg_x = F.avg_pool1d(padding_x, win_size, stride=1)
    if len(size) < 2:
        avg_x = avg_x.squeeze(0)
    return avg_x


def line_with_confidence():
    """
    draw acc vs cluster number with confidence (std)
    """
    fig, ax = plt.subplots()
    # mlp
    mean = np.array([57.06, 56.85, 57.28, 57.20, 56.87, 56.72])
    std = np.array([0.25, 0.39, 0.30, 0.12, 0.31, 0.24])
    x_label = [1, 2, 3, 5, 10, 50]
    y_min = 56
    fig_name = "mlp_arxiv_np"
    # nodeformer
    mean = np.array([65.54, 66.89, 65.62])
    std = np.array([0.78, 1.25, 0.72])
    x_label = [5, 10, 50]
    y_min = 64
    fig_name = "nodeformer_arxiv_np"
    # gcn
    mean = np.array([72.11, 72.08])
    std = np.array([0.31, 0.15])
    x_label = [10, 40]
    y_min = 71.5
    fig_name = "gcn_arxiv_np"
    # sage
    mean = np.array([72.29,  # pay attention
                     72.38, 72.15, 72.01, 72.16, 71.91, 71.92, 71.96])
    std = np.array([0.09,  # pay attention
                    0.03, 0.47, 0.33, 0.48, 0.13, 0.26, 0.22])
    x_label = [2, 3, 5, 10, 40, 70, 100, 150]
    y_min = 71
    fig_name = "sage_arxiv_np"

    x = [i for i in range(len(x_label))]
    low = mean - std
    high = mean + std

    ax.plot(mean)
    ax.fill_between(
        x, low, high, color='b', alpha=.15)
    ax.set_ylim(ymin=y_min)
    ax.set_xlabel("The number of clusters", fontsize=15)
    ax.set_ylabel("Accuracy %", fontsize=15)
    # ax.set_title('')
    plt.xticks(ticks=x, labels=x_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(fig_name)
    plt.show()


def cmp_cluster_feat_km():
    """
    draw dbi and ch value during training
    """
    stat_test = Analyzer.load_statistics("./analysis_data/test_analysis_arxiv_sage_np40eg199dp0.3wu500ep2000.pkl")
    stat_ablation = Analyzer.load_statistics(
        "./analysis_data/test_analysis_arxiv_sage_ablation_dp0.3ep1500run3knp40.pkl")

    baseline, result = stat_test["baseline"], stat_test["result"]
    baseline2, result2 = stat_ablation["baseline"], stat_ablation["result"]

    idx = 0
    y = np.array(result[idx], dtype=np.float32)
    y2 = np.array(result2[idx], dtype=np.float32)
    s, e = 0, 300

    # 1. dbi
    x = smooth_avg(y[:, -3][s:e])
    x_baseline = smooth_avg(y2[:, -3][s:e])
    file_name = "./figs/line_train_dbi.svg"
    name = 'with ClusterALL, k=5'
    baseline_name = 'original'
    xlabel_legend = "Training epochs"
    y_label_legend = "DBI value"
    name = '使用 ClusterALL, k=5'
    baseline_name = '原模型'
    xlabel_legend = "训练轮次"
    y_label_legend = "DBI 值"

    # 2. ch
    x = smooth_avg(y[:, -4][s:e]) / 1e5
    x_baseline = smooth_avg(y2[:, -4][s:e]) / 1e5
    file_name = "./figs/line_train_ch.svg"
    name = 'with ClusterALL, k=5'
    baseline_name = 'original'
    xlabel_legend = "Training epochs"
    y_label_legend = "CH value (x1e5)"
    name = '使用 ClusterALL, k=5'
    baseline_name = '原模型'
    xlabel_legend = "训练轮次"
    y_label_legend = "CH 值 (x1e5)"

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(x, label=name, color="blue")
    ax.plot(x_baseline, label=baseline_name, color="red")
    ax.set_ylabel(y_label_legend, fontsize=15)
    ax.legend(loc='upper right')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel(xlabel_legend, fontsize=15)
    plt.savefig(file_name, dpi=300, format="svg")
    plt.show()


def line_best():
    """
    draw acc vs cluster number k using mlp, nodeformer, gcn, graphsage
    """
    fig, ax = plt.subplots()
    # mlp
    y = np.array([57.45, 57.19, 57.71, 57.25, 57.37, 57.31, 57.01, 56.97, 56.86])
    y_min = 56
    baseline = 56.12
    fig_name = "figs/mlp_arxiv_best.svg"
    # nodeformer
    y = np.array([68.62, 68.77, 67.94, 68.49, 68.66, 67.62, 68.13, 67.72, 68.48])
    y_min = 67.5
    baseline = 67.92
    fig_name = "figs/nodeformer_arxiv_best.svg"
    # gcn
    y = np.array([72.28, 72.79, 72.85, 72.17, 72.54, 72.44, 72.54, 72.37, 72.39])
    y_min = 72
    baseline = 72.1
    fig_name = "figs/gcn_arxiv_best.svg"
    # sage
    y = np.array([72.52, 72.65, 72.4, 72.23, 72.64, 72.52, 72.1, 72.57, 72.44])
    y_min = 71.5
    baseline = 71.87
    fig_name = "figs/sage_arxiv_best.svg"

    x_label = [1, 2, 3, 4, 5, 10, 25, 50, 100]
    x = [i for i in range(len(x_label))]
    baseline = [baseline for _ in x_label]

    name = "with ClusterALL"
    baseline_label = "without ClusterALL"
    xlabel_legend = "The number of clusters k"
    ylabel_legend = "Accuracy %"
    name = "使用 ClusterALL"
    baseline_label = "原始模型"
    xlabel_legend = "聚类数量 k"
    ylabel_legend = "准确度 %"
    ax.plot(y, label=name)
    ax.plot(baseline, label=baseline_label, color="red", linestyle="--")
    ax.set_ylim(ymin=y_min)
    ax.set_xlabel(xlabel_legend, fontsize=15)
    ax.set_ylabel(ylabel_legend, fontsize=15)
    plt.xticks(ticks=x, labels=x_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(fig_name, dpi=300, format="svg")
    plt.show()


def line_best_update_gap():
    """
    draw accuracy vs update gap
    """
    fig, ax = plt.subplots()
    # update gap
    y = np.array([72.11, 72.04, 72.45, 72.63, 72.34])
    y_min = 71.5
    baseline = 72
    name = "with update"
    name = "有更新"
    fig_name = "figs/update_gap_arxiv_best.svg"

    x_label = [0, 19, 49, 199, 499]
    x = [i for i in range(len(x_label))]
    baseline = [baseline for _ in x_label]

    baseline_label = "No update"
    x_label_legend = "Update gap"
    y_label_legend = "Accuracy %"
    baseline_label = "无更新"
    x_label_legend = "更新间隔"
    y_label_legend = "准确度 %"
    ax.plot(y, label=name)
    ax.plot(baseline, label=baseline_label, color="red", linestyle="--")
    ax.set_ylim(ymin=y_min)
    ax.set_xlabel(x_label_legend, fontsize=15)
    ax.set_ylabel(y_label_legend, fontsize=15)
    plt.xticks(ticks=x, labels=x_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(fig_name, dpi=300, format="svg")
    plt.show()


def draw_graph_clusters(data: Data, pos: torch.Tensor, cluster_mapping: torch.Tensor,
                        name=None,
                        with_label: bool = False,
                        arrows=False, layout=None):
    """
    used by draw_cluster_stat
    """
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

    G = to_networkx(data)
    n = G.number_of_nodes()
    cluster_ids = torch.unique(cluster_mapping, return_counts=False)
    re_cluster_ids = torch.zeros((cluster_ids.max() + 1,), dtype=torch.long)
    re_cluster_ids[cluster_ids] = torch.arange(len(cluster_ids))

    color_map = list(map(lambda x: color(tuple(x)), ncolors(len(cluster_ids))))

    if layout == "spring":
        fix_pos = nx.spring_layout(G)
    elif layout == "random":
        fix_pos = nx.random_layout(G)
    else:
        fix_pos = dict(zip([i for i in range(n)], pos.tolist()))

    colors = [color_map[re_cluster_ids[i]] for i in cluster_mapping.tolist()]
    fig, ax = plt.subplots()
    nx.draw(G, pos=fix_pos, ax=ax, node_color=colors, arrows=arrows, with_labels=with_label, node_size=10,
            edge_color='grey')

    plt.savefig("./figs/clusters.svg" if name is None else f"./{name}", dpi=300, format='svg')
    plt.show()


def draw_cluster_stat():
    """
    visualizes ClusterALL's clustering infos
    """
    # get data
    stat = Analyzer.load_statistics("./analysis_data/test_analysis_arxiv_sage_np5eg199dp0.3wu0ep1500.4.pkl")
    best_x = stat["best_embed"]
    best_mapping = stat["best_mapping"]

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data/ogb/')
    data = dataset[0]
    y = data.y

    def find_alone_node(adj, node_num):
        edge_index_ = remove_self_loops(adj)[0]
        mask = torch.zeros(node_num)
        mask[edge_index_.reshape(-1, )] = 1
        alone_node = torch.where(mask == 0)[0]
        print(f"num of alone nodes: {len(alone_node)}, {alone_node}")
        return len(alone_node)

    N = data.x.size(0)
    cluster_ids, n_per_c = torch.unique(best_mapping, return_counts=True)
    print(f"cluster info: clusters: {cluster_ids}, nums: {n_per_c}")

    # 1. draw all clusters
    rand_idx = torch.arange(N)
    edge_index = torch.empty((2, 0))
    layout_name = None
    fig_name = "./figs/all_clusters.svg"

    # 2. draw a certain cluster
    # cluster_where = torch.where(best_mapping == 1)
    # rand_idx = cluster_where[0]
    # edge_index, _ = subgraph(rand_idx, data.edge_index, relabel_nodes=True)
    # layout_name = "random"
    # fig_name = "./figs/one_clusters.svg"
    # find_alone_node(edge_index, len(x))  # prompt

    # get necessary data
    x = best_x[0][rand_idx]
    y_sample = y[rand_idx]
    mapping = best_mapping[rand_idx]

    # prompt
    y_idx = torch.unique(y_sample)
    print(f"has {len(y_idx)} classes: {y_idx}")
    print(f"node num: {x.size(0)}, edge num: {edge_index.size(1)}")

    # draw
    data = Data(x=x, edge_index=edge_index)
    draw_graph_clusters(data, x, mapping, name=fig_name, layout=layout_name)


if __name__ == "__main__":
    draw_cluster_stat()
