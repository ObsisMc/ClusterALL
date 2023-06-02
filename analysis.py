import pickle
import os

import torch
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'


# you'd better put data for analysis into the directory 'analysis_data'
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
                         "best_ch": best_ch, "best_embed": best_embed, "best_mapping": self.mapping_best}, f)

    @staticmethod
    def load_statistics(save_path):
        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            stat = pickle.load(f)
        return stat


def main():
    """
    show analysis data during a certain training
    """
    # get analysis data
    stat = Analyzer.load_statistics("./analysis_data/test_analysis_arxiv_sage_np5eg199dp0.3wu0ep1500.4.pkl")
    baseline, result = stat["baseline"], stat["result"]
    y = np.array(result[0])
    train = y[:, 0]
    test = y[:, 2]
    dbi_km = y[:, -3]
    dbi = y[:, -1]
    ch_km = y[:, -4]
    ch = y[:, -2]

    # 1. plot training and testing lines
    # plt.plot(train, label="train")
    # plt.plot(test, label="test")

    # 2. plot dbi line
    # plt.plot([baseline[1] for _ in range(len(dbi))], label="dbi_bl")
    # plt.plot(dbi_km, label="dbi_km")
    # plt.plot(dbi, label="dbi")

    # 3. plot ch line
    # plt.plot([baseline[0] for _ in range(len(dbi))], label="ch_bl")
    # plt.plot(ch_km, label="ch_km")
    # plt.plot(ch, label="ch")

    # 4. plot two figs in one window
    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = ax.twinx()  # create the second axis
    ## 4.1 plot dbi line
    ax.plot([baseline[1] for _ in range(len(dbi))], label='bl', color="purple")
    ax.plot(dbi, label='dbi', color="blue")
    ax.plot(dbi_km, label='dbi_km', color="green")
    ax.legend(loc='upper left')
    ## 4.2 plot ch line
    # ax.plot([baseline[0] for _ in range(len(dbi))], label='bl', color="purple")
    # ax.plot(ch, label='ch', color="blue")
    # ax.plot(ch_km, label='ch_km', color="green")
    # ax.legend(loc='upper left')
    ## 4.3 plot training and testing line
    ax2.plot(train, label='train', color="orange")
    ax2.plot(test, label='test', color="red")
    ax2.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    main()
