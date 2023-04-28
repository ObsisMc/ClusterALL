import argparse

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from ogb_models.logger import Logger

from ogb_models.MLPCluster import MLPCluster, MLPClusterDataset, MLPClusterLoader, ClusterOptimizer


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, loader, train_idx, optimizer, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    # Modify
    data = loader[0]
    x, edge_index, y_true = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    out, infos, _ = model(x, edge_index)
    loader.update_cluster(infos[1])

    out, y_true = loader.convert(out), loader.convert(y_true)
    loss = criterion(out[train_idx], y_true[train_idx].to(torch.float))
    print(f'Loss: {loss:.4f}')
    cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
    print(f"cluster infos: {len(cluster_ids)} clusters, "
          f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, loader, split_idx, evaluator, device):
    model.eval()

    # Modify
    data = loader[0]
    x, edge_index, y_true = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    y_pred, infos, _ = model(x, edge_index)
    y_pred, y_true = loader.convert(y_pred), loader.convert(y_true)

    cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
    print(f"cluster infos: {len(cluster_ids)} clusters, "
          f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    # Modify
    parser.add_argument('--num_parts', type=int, default=5)
    parser.add_argument('--epoch_gap', type=int, default=99)
    parser.add_argument('--dropout_cluster', type=float, default=0.3)
    parser.add_argument('--warm_up', type=int, default=0)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins', root='../data/ogb/')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to('cpu')

    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)

    x = x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    # Modify: modify in_channel and out_channel
    model = MLP(args.hidden_channels, args.hidden_channels, args.hidden_channels, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    # Modify
    model = MLPCluster(model, x.size(-1), args.hidden_channels, 112, None, args.num_parts,
                       dropout=args.dropout_cluster).to(device)
    data = Data(x=x, y=y_true, edge_index=data.edge_index)
    dataset = MLPClusterDataset(dataset, data, split_idx, num_parts=args.num_parts)
    training_loader = MLPClusterLoader(dataset, "all", batch_size=-1, is_eval=False, shuffle=False)
    testing_loader = MLPClusterLoader(dataset, "all", batch_size=-1, is_eval=True, shuffle=False)

    for run in range(args.runs):
        # Modify
        model.reset_parameters(dataset.get_init_vnode(device))
        cluster_optimizer = ClusterOptimizer(model, args.epoch_gap, args.lr, args.warm_up)
        optimizer = torch.optim.Adam(cluster_optimizer.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            # Modify
            cluster_optimizer.zero_grad_step(epoch)

            loss = train(model, training_loader, train_idx, optimizer, device)  # Modify: pass loader

            if epoch % args.eval_steps == 0:
                result = test(model, testing_loader, split_idx, evaluator, device)  # Modify: pass dataset
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'\033[1;31m'
                          f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%'
                          f'\033[0m')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
