import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader
from torch_geometric.data import Data
from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, to_sparse_tensor
from eval import evaluate_cpu, eval_acc, eval_rocauc, eval_f1, evaluate_cpu_mini
from parse import parse_method, parser_add_main_args
import time
import math
import warnings


class TmpDataLoader:
    def __init__(self, data, idx, batch_size):
        self.idx = idx
        self.data = data
        self.n = self.data.x.size(0)
        self.batch_size = batch_size
        self.l = math.ceil(len(idx) / batch_size)

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration

        idx_i = self.idx[item * self.batch_size:(item + 1) * self.batch_size]
        x_i = self.data.x[idx_i]
        y_i = self.data.y[idx_i] if self.data.y is not None else None
        edge_index_i, _ = subgraph(idx_i, self.data.edge_index, num_nodes=self.n, relabel_nodes=True)
        data = Data(x=x_i, edge_index=edge_index_i, y=y_i)
        data.n_id = self.data.n_id[idx_i]
        return data


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

adjs = []
adj, _ = remove_self_loops(edge_index)
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1):  # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
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
    train_idx = split_idx['train'].to(device)
    if args.pre_trained:
        checkpoint_dir = f'../model/{args.dataset}-nodeformer.pkl'
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint)
    else:
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
    train_data = Data(x=train_x, y=train_label, edge_index=train_edge_index)
    train_data.n_id = torch.arange(n)
    training_loader = GraphSAINTRandomWalkSampler(train_data, batch_size=args.batch_size,
                                                  walk_length=3, num_steps=args.num_batchs,
                                                  sample_coverage=0)
    # training_loader = TmpDataLoader(data=train_data, batch_size=args.batch_size, idx=torch.arange(train_data.x.size(0)))

    for epoch in range(args.epochs):
        model.to(device)
        model.train()

        for i, sampled_data in enumerate(training_loader):
            x_i, label_i = sampled_data.x.to(device), sampled_data.y.to(device)
            n_id_i = sampled_data.n_id.to(device)
            adjs_i = []
            edge_index_i = sampled_data.edge_index.to(device)
            adjs_i.append(edge_index_i)
            for k in range(args.rb_order - 1):
                edge_index_i, _ = subgraph(n_id_i, adjs[k + 1], num_nodes=n, relabel_nodes=True)
                adjs_i.append(edge_index_i.to(device))
            optimizer.zero_grad()
            out_i, link_loss_ = model(x_i, adjs_i, args.tau)
            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i, label_i.squeeze(1).to(torch.float))
            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i, label_i.squeeze(1))
            loss -= args.lamda * sum(link_loss_) / len(link_loss_)
            print(f'Run: {run + 1:02d}/{args.runs:02d}, '
                  f'Epoch: {epoch:02d}/{args.epochs - 1:02d}, '
                  f'Batch: {i:02d}/{args.num_batchs - 1:02d}, '
                  f'Loss: {loss:.4f}')
            loss.backward()
            optimizer.step()
            if args.dataset == 'ogbn-proteins':
                scheduler.step()

        if epoch % 9 == 0:
            result = evaluate_cpu_mini(model, dataset, split_idx, eval_func, criterion, args)
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
