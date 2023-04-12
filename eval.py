import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from Clusteror import MyDataLoader, MyDataLoaderFC
from torch_geometric.data import Data
import tqdm

from loader import MyClusterData
from preprocess import get_adjs


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()
    if args.method == 'nodeformer':
        out, _ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
    else:
        out = model(dataset)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, num_parts: int = None, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    if num_parts is not None:
        data = Data(x=x, edge_index=adjs_[0])
        loader = MyDataLoader(data=data, num_parts=num_parts, batch_size=-1)
        sampled_data, mapping = loader[0]
        out, _, infos = model(sampled_data.x, mapping=mapping, adjs=[sampled_data.edge_index])
        print(infos[0], infos[1])
    else:
        adjs = []
        adjs.append(adjs_[0])
        for k in range(args.rb_order - 1):
            adjs.append(adjs_[k + 1])
        out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu_fc(model, dataset, split_idx, eval_func, criterion, args, num_parts: int = None, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    if num_parts is not None:
        data = Data(x=x, edge_index=adjs_[0])
        loader = MyDataLoaderFC(data=data, num_parts=num_parts, batch_size=-1, eval=True)
        sampled_data = loader[0]
        out, _, infos = model(sampled_data.x, mapping=None, adjs=[sampled_data.edge_index])
        print(infos[0], infos[1])
    else:
        adjs = []
        adjs.append(adjs_[0])
        for k in range(args.rb_order - 1):
            adjs.append(adjs_[k + 1])
        out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu_mini(model, dataset, split_idx, eval_func, criterion, args, num_parts=None, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph[
        'node_feat']
    data = Data(x, adjs_[0])
    data.n_id = torch.arange(x.size(0))

    split_names = ["valid", "test"]
    eval_nums = dict(zip(split_names, [split_idx[sn].size(0) for sn in split_names]))
    loaders = [(sn, NeighborLoader(data=data, num_neighbors=[-1, 10], input_nodes=split_idx[sn],
                                   batch_size=en, shuffle=False)) for sn, en in eval_nums.items()]
    outs = {"train": None, "valid": None, "test": None}
    print("begin eval model")
    # print(data.n_id[split_idx[split_name]][:20],data.n_id[split_idx[split_name]][-20:])
    for s_name, loader in loaders:
        for sampled_data in loader:
            # print(sampled_data.n_id[:test_num][:20],sampled_data.n_id[:test_num][-20:], sampled_data.n_id[:test_num+10][-20:])
            # print(sampled_data.x.size(0))
            if num_parts is not None:
                sampled_data, mapping = MyDataLoader(data=sampled_data, num_parts=num_parts, batch_size=-1)[0]
                outs[s_name], _ = model(sampled_data.x, mapping=mapping, adjs=[sampled_data.edge_index], )
            else:
                outs[s_name], _ = model(sampled_data.x, [sampled_data.edge_index])
    print("finish eval model")
    accs = {"train": None, "valid": None, "test": None}
    for key, item in outs.items():
        if item is None:
            accs[key] = 0
        else:
            accs[key] = eval_func(dataset.label[split_idx[key]], outs[key][:eval_nums[key]])
    # train_acc = eval_func(
    #     dataset.label[split_idx['train']], out[split_idx['train']])
    # train_acc = 0
    # valid_acc = eval_func(
    #     dataset.label[split_idx[]], outs)
    # test_acc = eval_func(
    #     dataset.label[split_idx['test']], out[split_idx['test']])
    # test_acc = 0
    valid_out = outs["valid"][:eval_nums["valid"]]
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(valid_out, true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(valid_out, dim=1)
        valid_loss = criterion(
            out, dataset.label.squeeze(1)[split_idx['valid']])

    return accs["train"], accs["valid"], accs["test"], valid_loss, outs


@torch.no_grad()
def evaluate_cpu_mini_fc(model, dataset, split_idx, eval_func, criterion, args, num_parts=None, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph[
        'node_feat']
    data = Data(x, adjs_[0])
    data.n_id = torch.arange(x.size(0))

    split_names = ["valid", "test"]
    eval_nums = dict(zip(split_names, [split_idx[sn].size(0) for sn in split_names]))
    loaders = [(sn, NeighborLoader(data=data, num_neighbors=[-1, 10], input_nodes=split_idx[sn],
                                   batch_size=en, shuffle=False)) for sn, en in eval_nums.items()]
    outs = {"train": None, "valid": None, "test": None}
    print("begin eval model")
    # print(data.n_id[split_idx[split_name]][:20],data.n_id[split_idx[split_name]][-20:])
    for s_name, loader in loaders:
        for sampled_data in loader:
            # print(sampled_data.n_id[:test_num][:20],sampled_data.n_id[:test_num][-20:], sampled_data.n_id[:test_num+10][-20:])
            # print(sampled_data.x.size(0))
            if num_parts is not None:
                sampled_data = MyDataLoaderFC(data=sampled_data, num_parts=num_parts, batch_size=-1, eval=True)[0]
                outs[s_name], _, infos = model(sampled_data.x, mapping=None, adjs=[sampled_data.edge_index], )
                print(infos[0], infos[1])
            else:
                outs[s_name], _ = model(sampled_data.x, [sampled_data.edge_index])
    print("finish eval model")
    accs = {"train": None, "valid": None, "test": None}
    for key, item in outs.items():
        if item is None:
            accs[key] = 0
        else:
            accs[key] = eval_func(dataset.label[split_idx[key]], outs[key][:eval_nums[key]])
    # train_acc = eval_func(
    #     dataset.label[split_idx['train']], out[split_idx['train']])
    # train_acc = 0
    # valid_acc = eval_func(
    #     dataset.label[split_idx[]], outs)
    # test_acc = eval_func(
    #     dataset.label[split_idx['test']], out[split_idx['test']])
    # test_acc = 0
    valid_out = outs["valid"][:eval_nums["valid"]]
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(valid_out, true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(valid_out, dim=1)
        valid_loss = criterion(
            out, dataset.label.squeeze(1)[split_idx['valid']])

    return accs["train"], accs["valid"], accs["test"], valid_loss, outs


@torch.no_grad()
def evaluate_cpu_cluster(model, dataset, split_idx, eval_func, criterion, args, result=None):
    # get testing data
    dataset.label = dataset.label.to("cpu")
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    data = Data(x, adjs_[0])
    data.n_id = torch.arange(x.size(0))
    valid_num = split_idx["valid"].size(0)
    loader = NeighborLoader(data=data, num_neighbors=[-1],
                            input_nodes=split_idx["valid"],
                            batch_size=valid_num, shuffle=False)
    # preprocess data
    cluster_loaders = []
    v_adjs = []
    pbar = tqdm.tqdm(desc="Testing clustering", total=len(loader))
    for pi, sampled_data in enumerate(loader):
        x_pi, edge_index_pi = sampled_data.x.to("cpu"), sampled_data.edge_index.to("cpu")
        n_id_pi = sampled_data.n_id.to("cpu")
        data_pi = Data(x=x_pi, edge_index=edge_index_pi)
        data_pi.n_id = n_id_pi
        data_pi.n_clst_id = torch.arange(x_pi.size(0))
        cluster_data_pi = MyClusterData(data=data_pi, num_parts=args.num_parts, recursive=False, log=False)
        cluster_loader_pi = ClusterLoader(cluster_data_pi, batch_size=1, shuffle=False)
        # vnode
        num_v_nodes_pi = cluster_data_pi.v_graph_num_nodes
        v_edge_index_pi, v_edge_attr_pi = cluster_data_pi.virtual_graph
        v_adjs_pi = get_adjs(v_edge_index_pi, args.rb_order, num_v_nodes_pi) if v_edge_attr_pi.size(0) else None

        cluster_loaders.append(cluster_loader_pi)
        v_adjs.append(v_adjs_pi)
        pbar.update()
    time.sleep(0.05)
    testing_loader = list(zip(cluster_loaders, v_adjs))

    out = None
    model.eval()
    model.to("cpu")
    pbar = tqdm.tqdm(desc="Testing", total=len(testing_loader))
    for i, (loader_i, v_adjs_i) in enumerate(testing_loader):
        out, _ = model(loader_i, v_adjs_i)
        pbar.update()
    time.sleep(0.05)

    # train_acc = eval_func(
    #     dataset.label[split_idx['train']], out[split_idx['train']])
    train_acc = 0
    out = out[:valid_num]
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out)
    # test_acc = eval_func(
    #     dataset.label[split_idx['test']], out[split_idx['test']])
    test_acc = 0
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out, true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out, dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
