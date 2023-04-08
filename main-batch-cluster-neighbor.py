import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler, ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.data import Data
from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, to_sparse_tensor
from eval import evaluate_cpu, eval_acc, eval_rocauc, eval_f1, evaluate_cpu_mini, evaluate_cpu_cluster
from parse_cluster import parse_method, parser_add_main_args
import time
import tqdm
from Clusteror import Clusteror, MyDataLoader

import warnings

warnings.filterwarnings('ignore')


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = "cpu"
else:
    device = "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
# device = "cpu"

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, dataset=args.dataset, protocol=args.protocol)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

print(f"num nodes {n} | num edges {edge_index.size(1)} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args, dataset, n, c, d, device)
model = Clusteror(encoder=model, in_channels=d, hidden_channels=args.hidden_channels, num_parts=args.num_parts,
                  out_channels=c, num_layers=args.num_layers, use_jk=args.use_jk).to(device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)


def get_adjs(e_idx, rb_order_inner, n_inner, device_inner="cpu"):
    adjs_inner = []
    adj_inner, _ = remove_self_loops(e_idx)
    adj_inner, _ = add_self_loops(adj_inner, num_nodes=n_inner)
    adjs_inner.append(adj_inner.to(device_inner))
    # for _ in range(rb_order_inner - 1):  # edge_index of high order adjacency
    #     adj_inner = adj_mul(adj_inner, adj_inner, n_inner)
    #     adjs_inner.append(adj_inner)
    return adjs_inner


adjs = get_adjs(edge_index, args.rb_order, n)
dataset.graph['adjs'] = adjs
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)
    best_val = float('-inf')

    # labels
    train_label = None
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        train_label = true_label[train_idx]
    else:
        train_label = dataset.label[train_idx]

    # get training data
    train_x = x[train_idx]
    train_edge_index, _ = subgraph(train_idx, adjs[0], num_nodes=n, relabel_nodes=True)

    # pre-processing
    train_data = Data(x=train_x, y=train_label, edge_index=train_edge_index)
    training_loader = MyDataLoader(data=train_data, batch_size=args.batch_size, num_parts=args.num_parts)
    num_batches = len(training_loader)

    # training
    for epoch in range(args.epochs):
        model.to(device)
        model.train()
        for i, (sampled_data, mapping) in enumerate(training_loader):
            optimizer.zero_grad()

            edge_index_test = train_data.edge_index
            src_idx = torch.where(edge_index_test[1] == 0)[0]
            src = edge_index_test[0][src_idx]

            x_i, edge_index_i = sampled_data.x.to(device), sampled_data.edge_index.to(device)
            out_i, link_loss_ = model(x_i, [edge_index_i], tau=args.tau, mapping=mapping)

            label_i = sampled_data.y.to(device)

            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i, label_i.squeeze(1).float())
            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i, label_i.squeeze(1))
            loss -= args.lamda * sum(link_loss_) / len(link_loss_)
            print(f'Run: {run + 1:02d}/{args.runs:02d}, '
                  f'Epoch: {epoch:02d}/{args.epochs - 1:02d}, '
                  f'Batch: {i:02d}/{num_batches - 1:02d}, '
                  f'Loss: {loss:.4f}')
            loss.backward()
            optimizer.step()
            if args.dataset == 'ogbn-proteins':
                scheduler.step()

        if epoch % 9 == 0:
            result = evaluate_cpu_mini(model, dataset, split_idx, eval_func, criterion, args, num_parts=args.num_parts)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            print(f'\033[1;31mEpoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%\033[0m')
    logger.print_statistics(run)

results = logger.print_statistics()
