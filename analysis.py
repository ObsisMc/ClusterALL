import pandas as pd
import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans
import pickle
import matplotlib.pyplot as plt
import os


class Analyzer:
    def __init__(self, runs, X, num_parts):
        self.results = [[] for _ in range(runs)]

        self.kmeans = MiniBatchKMeans(num_parts, batch_size=1000)
        self.ch_bl, self.dbi_bl = self.__get_cluster_baseline(X, self.kmeans)

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
        y_pred = self.kmeans.fit_predict(X)
        ch_bl = calinski_harabasz_score(X, y_pred)
        dbi_bl = davies_bouldin_score(X, y_pred)

        cluster_ids, n_per_c = torch.unique(mapping, return_counts=True)
        if len(cluster_ids) == 1:
            ch = dbi = None
        else:
            ch = calinski_harabasz_score(X, mapping)
            dbi = davies_bouldin_score(X, mapping)
        result = [*result[:3], ch_bl, dbi_bl, ch, dbi]

        self.results[run].append(result)

    def save_statistics(self, save_path):
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        with open(save_path, "wb") as f:
            pickle.dump({"baseline": [self.ch_bl, self.dbi_bl], "result": self.results}, f)

    @staticmethod
    def load_statistics(save_path):
        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            stat = pickle.load(f)
        return stat


def main():
    stat = Analyzer.load_statistics("../model/ogbn-arxiv/mlp/test_drop_eg0.pkl")
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




if __name__ == "__main__":
    main()
