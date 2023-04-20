import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from NodeformerCluster import NodeformerClusterLoader


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
def evaluate_cpu_cluster(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))

    loader = NodeformerClusterLoader(dataset, "all", batch_size=-1, is_eval=True, shuffle=False)
    sampled_data = loader[0]
    edge_mask_eval = [dataset.N_train__ * 2 + dataset.num_parts__,
                      dataset.N_train__] if dataset.num_parts__ > 0 else None
    out, infos, _ = model(sampled_data.x, sampled_data.edge_index,
                          adjs=[sampled_data.edge_index],
                          edge_mask=edge_mask_eval)
    cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
    print(f"cluster infos: {len(cluster_ids)} clusters, "
          f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")

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
def evaluate_cpu_mini_cluster(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))

    split_names = ["train"]  # can only be 'train', 'valid', 'test'
    outs = {"train": None, "valid": None, "test": None}
    print("begin eval model")
    edge_mask_eval = [dataset.N_train__ * 2 + dataset.num_parts__,
                      dataset.N_train__] if dataset.num_parts__ > 0 else None
    for s_name in split_names:
        sampled_data= NodeformerClusterLoader(dataset, s_name, batch_size=-1, is_eval=True, shuffle=False)[0]
        outs[s_name], infos, _ = model(sampled_data.x, sampled_data.edge_index,
                                       adjs=[sampled_data.edge_index],
                                       edge_mask=edge_mask_eval)
        cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
        print(f"cluster infos: {len(cluster_ids)} clusters, "
              f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")
    print("finish eval model")
    accs = {"train": None, "valid": None, "test": None}
    for key, item in outs.items():
        if item is None:
            accs[key] = 0
        else:
            if key == "valid" or key == "test":
                accs[key] = eval_func(dataset.label[split_idx[key]], outs[key][dataset.N_train__:])
            else:
                accs[key] = eval_func(dataset.label[split_idx[key]], outs[key])

    if outs["valid"] is not None:
        valid_loss = eval_loss(outs, split_idx, "valid", args.dataset, dataset, criterion)
    else:
        valid_loss = eval_loss(outs, split_idx, "train", args.dataset, dataset, criterion)

    return accs["train"], accs["valid"], accs["test"], valid_loss, outs


def eval_loss(result, split_idx, split_name, dataset_name, dataset, criterion):
    out = result[split_name]
    if out is None:
        return 0
    if dataset_name in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        loss = criterion(out, true_label.squeeze(1)[
            split_idx[split_name]].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out, dataset.label.squeeze(1)[split_idx[split_name]])

    return loss
