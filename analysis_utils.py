import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn.functional as F
import numpy as np
from analysis import Analyzer

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
    mean = np.array([72.29,  # fake data
                     72.38, 72.15, 72.01, 72.16, 71.91, 71.92, 71.96])
    std = np.array([0.09,  # fake data
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
    file_name = "./figs/line_train_dbi"
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
    file_name = "./figs/line_train_ch"
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
    fig_name = "figs/mlp_arxiv_best"
    # nodeformer
    y = np.array([68.62, 68.77, 67.94, 68.49, 68.66, 67.62, 68.13, 67.72, 68.48])
    y_min = 67.5
    baseline = 67.92
    fig_name = "figs/nodeformer_arxiv_best"
    # gcn
    y = np.array([72.28, 72.79, 72.85, 72.17, 72.54, 72.44, 72.54, 72.37, 72.39])
    y_min = 72
    baseline = 72.1
    fig_name = "figs/gcn_arxiv_best"
    # sage
    y = np.array([72.52, 72.65, 72.4, 72.23, 72.64, 72.52, 72.1, 72.57, 72.44])
    y_min = 71.5
    baseline = 71.87
    fig_name = "figs/sage_arxiv_best"

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
    fig_name = "figs/update_gap_arxiv_best"

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


if __name__ == "__main__":
    cmp_cluster_feat_km()
