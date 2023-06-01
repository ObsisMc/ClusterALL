import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from ogb_models.logger import Logger

from ogb_models.GNNCluster import GNNCluster, GNNClusterDataset, GNNClusterLoader, ClusterOptimizer
from analysis import Analyzer


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x  # Modify


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x  # Modify


# Modify
def train(model, loader, train_idx, optimizer, device):
    model.train()

    optimizer.zero_grad()

    data = loader[0].to(device)
    out, infos, _ = model(data.x, data.adj_t)
    loader.update_cluster(infos[1])
    out = F.log_softmax(out, dim=-1)
    out, y = loader.convert(out), loader.convert(data.y)
    loss = F.nll_loss(out[train_idx], y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
    print(f"cluster infos: {len(cluster_ids)} clusters, "
          f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")
    return loss.item()


# Modify
@torch.no_grad()
def test(model, loader, split_idx, evaluator, device):
    model.eval()

    data = loader[0].to(device)

    out, infos, _ = model(data.x, data.adj_t)
    out = F.log_softmax(out, dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_pred, y_true = loader.convert(y_pred), loader.convert(data.y)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, loader.convert(infos[3]), loader.convert(infos[1])


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--num_parts', type=int, default=5)
    parser.add_argument('--epoch_gap', type=int, default=99)
    parser.add_argument('--dropout_cluster', type=float, default=0.3)
    parser.add_argument('--warm_up', type=int, default=0)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor(),
                                     root='../data/ogb/')

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    # Modify
    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    # Modify
    model = GNNCluster(model, data.num_features, args.hidden_channels, dataset.num_classes, None,
                       num_parts=args.num_parts, dropout=args.dropout_cluster).to(device)
    analyzer = Analyzer(args.runs, data.x, args.num_parts)
    for run in range(args.runs):
        # Modify
        dataset = GNNClusterDataset(dataset, data, split_idx, num_parts=args.num_parts)
        training_loader = GNNClusterLoader(dataset, "all", is_eval=False, batch_size=-1, shuffle=False)
        testing_loader = GNNClusterLoader(dataset, "all", is_eval=True, batch_size=-1, shuffle=False)

        model.reset_parameters(dataset.get_init_vnode(device))

        cluster_optimizer = ClusterOptimizer(model, args.epoch_gap, args.lr, args.warm_up)
        optimizer = torch.optim.Adam(cluster_optimizer.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            # Modify
            cluster_optimizer.zero_grad_step(epoch)
            loss = train(model, training_loader, train_idx, optimizer, device)
            result = test(model, testing_loader, split_idx, evaluator, device)
            logger.add_result(run, result[:3])
            analyzer.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result[:3]
                print(f'\033[1;31m'
                      f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%'
                      f'\033[0m')

        logger.print_statistics(run)
    logger.print_statistics()

    analysis_name = f"./test_analysis_arxiv_{'sage' if args.use_sage else 'gcn'}_"
    config_list = [("num_parts", "np"), ("epoch_gap", "eg"), ("dropout_cluster", "dp"), ("warm_up", "wu"),
                   ("epochs", "ep")]
    for c in config_list:
        analysis_name += c[1]
        analysis_name += str(getattr(args, c[0], "None"))
    analysis_name += ".pkl"
    analyzer.save_statistics(analysis_name)


if __name__ == "__main__":
    main()
