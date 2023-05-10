import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans
import pickle
import matplotlib.pyplot as plt
import os
import utils
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, remove_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F


class Analyzer:
    def __init__(self, runs, X, num_parts, kmeans_num_parts=0):
        self.results = [[] for _ in range(runs)]
        self.num_parts = num_parts
        self.kmeans_num_parts = kmeans_num_parts
        self.embedding_best = self.embedding_k_best = None
        self.mapping_best = None
        self.ch_k_best = self.ch_best = 0
        self.dbi_k_best = self.dbi_best = torch.inf
        if num_parts > 0:
            self.kmeans = MiniBatchKMeans(num_parts, batch_size=500)
            self.ch_bl, self.dbi_bl = self.__get_cluster_baseline(X, self.kmeans)
        elif kmeans_num_parts > 0:
            self.kmeans = MiniBatchKMeans(kmeans_num_parts, batch_size=500)
            self.ch_bl, self.dbi_bl = self.__get_cluster_baseline(X, self.kmeans)
        else:
            self.kmeans = None
            self.ch_bl = self.dbi_bl = 0

    def __get_cluster_baseline(self, X, clusterer):
        X = X.to("cpu")

        y_pred = clusterer.fit_predict(X)
        ch_bl = calinski_harabasz_score(X, y_pred)
        dbi_bl = davies_bouldin_score(X, y_pred)
        return ch_bl, dbi_bl

    def add_result(self, run, result):
        assert len(result) == 5
        assert run >= 0 and run < len(self.results)
        X, mapping = result[-2:]
        X, mapping = X.to("cpu"), mapping.to("cpu")
        ch_bl = dbi_bl = ch = dbi = 0
        if self.num_parts > 0:
            y_pred = self.kmeans.fit_predict(X)
            ch_bl = calinski_harabasz_score(X, y_pred)
            dbi_bl = davies_bouldin_score(X, y_pred)
            if dbi_bl < self.dbi_k_best:
                self.embedding_k_best = X
            self.ch_k_best = ch_bl if ch_bl > self.ch_k_best else self.ch_k_best
            self.dbi_k_best = dbi_bl if dbi_bl < self.dbi_k_best else self.dbi_k_best

            cluster_ids, n_per_c = torch.unique(mapping, return_counts=True)
            if len(cluster_ids) == 1:
                ch = dbi = None
            else:
                ch = calinski_harabasz_score(X, mapping)
                dbi = davies_bouldin_score(X, mapping)
                if dbi < self.dbi_best:
                    self.embedding_best = X
                    self.mapping_best = mapping
                self.ch_best = ch if ch > self.ch_best else self.ch_best
                self.dbi_best = dbi if dbi < self.dbi_best else self.dbi_best

        elif self.kmeans_num_parts > 0:
            y_pred = self.kmeans.fit_predict(X)
            ch_bl = calinski_harabasz_score(X, y_pred)
            dbi_bl = davies_bouldin_score(X, y_pred)
            if dbi_bl < self.dbi_k_best:
                self.embedding_k_best = X
            self.ch_k_best = ch_bl if ch_bl > self.ch_k_best else self.ch_k_best
            self.dbi_k_best = dbi_bl if dbi_bl < self.dbi_k_best else self.dbi_k_best

        result = [*result[:3], ch_bl, dbi_bl, ch, dbi]

        self.results[run].append(result)

    def save_statistics(self, save_path):
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        best_dbi = [self.dbi_best, self.dbi_k_best]
        best_ch = [self.ch_best, self.ch_k_best]
        best_embed = [self.embedding_best, self.embedding_k_best]
        with open(save_path, "wb") as f:
            pickle.dump({"baseline": [self.ch_bl, self.dbi_bl], "result": self.results, "best_dbi": best_dbi,
                         "best_ch": best_ch, "best_embed":best_embed, "best_mapping": self.mapping_best}, f)

    @staticmethod
    def load_statistics(save_path):
        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            stat = pickle.load(f)
        return stat


def main():
    stat = Analyzer.load_statistics("./test_analysis_arxiv_sage_np40eg199dp0.4wu0ep1500attn2_2.pkl")
    baseline, result = stat["baseline"], stat["result"]

    y = np.array(result[0])
    train = y[:, 0]
    test = y[:, 2]
    dbi_km = y[:, -3]
    dbi = y[:, -1]
    ch_km = y[:, -4]
    ch = y[:, -2]
    # plt.plot(train, label="train")
    # plt.plot(test, label="test")

    # plt.plot([baseline[1] for _ in range(len(dbi))], label="dbi_bl")
    # plt.plot(dbi_km, label="dbi_km")
    # plt.plot(dbi, label="dbi")

    # plt.plot([baseline[0] for _ in range(len(dbi))], label="ch_bl")
    # plt.plot(ch_km, label="ch_km")
    # plt.plot(ch, label="ch")

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot([baseline[1] for _ in range(len(dbi))], label='bl', color="purple")
    ax.plot(dbi, label='dbi', color="blue")
    ax.plot(dbi_km, label='dbi_km', color="green")

    # ax.plot([baseline[0] for _ in range(len(dbi))], label='bl', color="purple")
    # ax.plot(ch, label='ch', color="blue")
    # ax.plot(ch_km, label='ch_km', color="green")
    # ax.set_ylabel("℃")
    # ax.set_ylim(0, 40)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(train, label='train', color="orange")
    ax2.plot(test, label='test', color="red")
    # ax2.set_ylabel("℉")
    # ax2.set_ylim(60, 100)
    ax2.legend(loc='upper right')

    plt.show()


def main_cmp():
    stat = Analyzer.load_statistics("./test_analysis_arxiv_sage_np40eg199dp0.3wu500ep2000.pkl")
    stat2 = Analyzer.load_statistics("./test_analysis_arxiv_sage_ablation_dp0.3ep1500run3knp40.pkl")

    baseline, result = stat["baseline"], stat["result"]
    baseline2, result2 = stat2["baseline"], stat2["result"]
    # print(stat["best_dbi"])
    # print(stat["best_embed"])
    # print(stat["best_mapping"])

    train = np.mean(np.array(result)[:, :, 0], axis=0)
    test = np.mean(np.array(result)[:, :, 2], axis=0)
    train2 = np.mean(np.array(result2)[:, :, 0], axis=0)
    test2 = np.mean(np.array(result2)[:, :, 2], axis=0)

    idx = 0
    y = np.array(result[idx])
    y2 = np.array(result2[idx])
    s, e = 0, 1000
    dbi_km = y[:, -3][s:e]
    dbi = y[:, -1][s:e]
    dbi_km2 = y2[:, -3][s:e]
    dbi2 = y2[:, -1][s:e]
    ch_km = y[:, -4][s:e]
    ch = y[:, -2][s:e]
    ch_km2 = y2[:, -4][s:e]
    ch2 = y2[:, -2][s:e]

    # smooth
    # dbi_km = smooth_avg(np.array(dbi_km, dtype=np.float32))
    # dbi = smooth_avg(numpy.array(padding_head_nan(dbi), dtype=np.float32))
    # dbi_km2 = smooth_avg(np.array(dbi_km2, dtype=np.float32))

    # train_avg = torch.mean(result[:,:,0], dim=0)
    # plt.plot(train, label="train")
    # plt.plot(test, label="test")

    # plt.plot([baseline[1] for _ in range(len(dbi))], label="dbi_bl")
    # plt.plot(dbi_km, label="dbi_km")
    # plt.plot(dbi, label="dbi")

    # plt.plot([baseline[0] for _ in range(len(dbi))], label="ch_bl")
    # plt.plot(ch_km, label="ch_km")
    # plt.plot(ch, label="ch")

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot([baseline[1] for _ in range(e - s)], label='bl', color="purple")
    ax.plot(dbi, label='dbi', color="cyan")
    ax.plot(dbi_km, label='dbi_km', color="green")
    # ax.plot(dbi2, label='dbi2', color="black")
    ax.plot(dbi_km2, label='dbi_km2', color="grey")

    # ax.plot([baseline[0] for _ in range(e-s)], label='bl', color="purple")
    # ax.plot(ch, label='ch', color="blue")
    # ax.plot(ch_km, label='ch_km', color="green")
    # ax.plot(dbi2, label='dbi2', color="black")
    # ax.plot(ch_km2, label='ch_km2', color="grey")
    # ax.set_ylabel("℃")
    # ax.set_ylim(0, 40)
    ax.legend(loc='upper left')

    # ax2 = ax.twinx()
    # ax2.plot(train, label='train', color="orange")
    # ax2.plot(test, label='test', color="red")
    # ax2.plot(train2, label='train', color="blue")
    # ax2.plot(test2, label='test', color="purple")
    # ax2.set_ylabel("℉")
    # ax2.set_ylim(60, 100)
    # ax2.legend(loc='upper right')

    plt.show()


def smooth_avg(x, win_size=10):
    x = torch.tensor(x, dtype=torch.float)
    size = x.size()
    if len(size) < 2:
        x = x.reshape(1, -1)
    assert len(x.size()) == 2

    padding_x = torch.cat([x[:, 0].reshape(-1, 1).repeat(1, win_size - 1), x], dim=1)
    avg_x = F.avg_pool1d(padding_x, win_size, stride=1)
    if len(size) < 2:
        avg_x = avg_x.squeeze(0)
    return avg_x


def padding_head_nan(x):
    """
    only the first several values are None
    """
    none_mask = x == None
    x[none_mask] = x[np.sum(none_mask)]
    return x


def main_cmp_cluster_feat_km():
    stat_test = Analyzer.load_statistics("./test_analysis_arxiv_sage_np40eg199dp0.3wu500ep2000.pkl")
    stat_ablation = Analyzer.load_statistics("./test_analysis_arxiv_sage_ablation_dp0.3ep1500run3knp40.pkl")

    baseline, result = stat_test["baseline"], stat_test["result"]
    baseline2, result2 = stat_ablation["baseline"], stat_ablation["result"]

    idx = 0
    y = np.array(result[idx], dtype=np.float32)
    y2 = np.array(result2[idx], dtype=np.float32)
    s, e = 0, 500
    dbi_km = smooth_avg(y[:, -3][s:e])
    dbi_km2 = smooth_avg(y2[:, -3][s:e])
    ch_km = smooth_avg(y[:, -4][s:e])
    ch_km2 = smooth_avg(y2[:, -4][s:e])

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(dbi_km, label='DBI with FE', color="green")
    ax.plot(dbi_km2, label='original DBI', color="blue")
    ax.set_ylabel("DBI value", fontsize=15)
    # ax.set_ylim(60, 100)
    ax.legend(loc='upper right')

    ax2 = ax.twinx()
    ax2.plot(ch_km / 1e5, label='CH value with FE', color="orange")
    ax2.plot(ch_km2 / 1e5, label='original CH value', color="red")
    ax2.set_ylabel("CH value (x1e5)", fontsize=15)
    # ax2.set_ylim(60, 100)
    ax2.legend(loc='upper center')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel("Training epochs", fontsize=15)
    plt.savefig("./line_train_dbi&ch")
    plt.show()


def main_draw():
    stat = Analyzer.load_statistics("./test_analysis_arxiv_sage_np40eg199dp0.4wu0ep1500attn2_2.pkl")

    baseline, result = stat["baseline"], stat["result"]
    print(stat["best_dbi"])
    print(stat["best_embed"])
    print(stat["best_mapping"])

    train = np.mean(np.array(result)[:, :, 0], axis=0)
    test = np.mean(np.array(result)[:, :, 2], axis=0)

    idx = 0
    y = np.array(result[idx])
    s, e = 0, 1000
    dbi_km = y[:, -3][s:e]
    dbi = y[:, -1][s:e]
    ch_km = y[:, -4][s:e]
    ch = y[:, -2][s:e]

    best_dbi = stat["best_dbi"]
    best_x = stat["best_embed"]
    best_mapping = stat["best_mapping"]

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data/ogb/')
    data = dataset[0]

    def find_alone_node(adj, node_num):
        edge_index_ = remove_self_loops(adj)[0]
        mask = torch.zeros(node_num)
        mask[edge_index_.reshape(-1, )] = 1
        alone_node = torch.where(mask == 0)[0]
        print(f"num of alone nodes: {len(alone_node)}, {alone_node}")

    N = data.x.size(0)
    find_alone_node(data.edge_index, N)

    sample_num = data.x.size(0)
    rand_idx = torch.randint(len(data.x), size=(sample_num,))
    rand_idx = torch.tensor(sorted(list(set(rand_idx.tolist()))), dtype=torch.long)

    rand_idx = torch.arange(1000, 2000)

    cluster0_where = torch.where(best_mapping == best_mapping[2])
    rand_idx = cluster0_where[0]
    rand_idx = torch.arange(N)
    x = best_x[0][rand_idx]
    edge_index, _ = subgraph(rand_idx, data.edge_index, relabel_nodes=True)
    find_alone_node(edge_index, len(x))
    mapping = best_mapping[rand_idx]

    print(f"node num: {x.size(0)}, edge num: {edge_index.size(1)}")
    data = Data(x=x, edge_index=edge_index)
    utils.draw_graph_clusters(data, x, mapping)

    # smooth
    # dbi_km = smooth_avg(np.array(dbi_km, dtype=np.float32))
    # dbi = smooth_avg(numpy.array(padding_head_nan(dbi), dtype=np.float32) )
    # dbi_km2 = smooth_avg(np.array(dbi_km2, dtype=np.float32))

    # train_avg = torch.mean(result[:,:,0], dim=0)
    # plt.plot(train, label="train")
    # plt.plot(test, label="test")

    # plt.plot([baseline[1] for _ in range(len(dbi))], label="dbi_bl")
    # plt.plot(dbi_km, label="dbi_km")
    # plt.plot(dbi, label="dbi")

    # plt.plot([baseline[0] for _ in range(len(dbi))], label="ch_bl")
    # plt.plot(ch_km, label="ch_km")
    # plt.plot(ch, label="ch")

    # fig = plt.figure()
    # ax = fig.add_subplot()
    #
    # ax.plot([baseline[1] for _ in range(e - s)], label='bl', color="purple")
    # ax.plot(dbi, label='dbi', color="cyan")
    # ax.plot(dbi_km, label='dbi_km', color="green")
    #
    # # ax.plot([baseline[0] for _ in range(e-s)], label='bl', color="purple")
    # # ax.plot(ch, label='ch', color="blue")
    # # ax.plot(ch_km, label='ch_km', color="green")
    # # ax.plot(dbi2, label='dbi2', color="black")
    # # ax.plot(ch_km2, label='ch_km2', color="grey")
    # # ax.set_ylabel("℃")
    # # ax.set_ylim(0, 40)
    # ax.legend(loc='upper left')
    #
    # plt.show()


if __name__ == "__main__":
    main_cmp_cluster_feat_km()
