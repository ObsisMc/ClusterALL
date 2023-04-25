import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from ogb_models.logger import Logger

from ogb_models.GNNCluster import GNNCluster, GNNClusterDataset, GNNClusterLoader
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def train(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    # Modify
    data = loader[0]
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out, data.y.to(torch.float))

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, dataset, split_idx, evaluator):
    model.eval()
    model.to("cpu")

    testing_loader = GNNClusterLoader(dataset, "all", is_eval=True, batch_size=-1, shuffle=False)
    data = testing_loader[0]
    x, y, edge_index = data.x.to("cpu"), data.y.to("cpu"), data.edge_index.to("cpu")
    y_pred = model(x, edge_index)
    data.y, y_pred = testing_loader.convert(y), testing_loader.convert(y_pred)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=0)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=9)
    parser.add_argument('--runs', type=int, default=10)
    # Modify
    parser.add_argument('--num_parts', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2000)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'), root='../../data/ogb/')
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    # Modify
    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    # Modify
    data = data

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    # Modify
    model = GNNCluster(model, data.num_features, args.hidden_channels, 112, None, args.num_parts).to(device)
    row, col, _ = data.adj_t.t().coo()
    edge_index = torch.stack([row, col])
    data = Data(x=data.x, y=data.y, edge_index=edge_index)
    dataset = GNNClusterDataset(dataset, data, split_idx, num_parts=args.num_parts)
    training_loader = GNNClusterLoader(dataset, "train", is_eval=False, batch_size=args.batch_size, shuffle=False)
    batch_num = len(training_loader)
    criterion = torch.nn.BCEWithLogitsLoss()

    for run in range(args.runs):
        model.reset_parameters(dataset.get_init_vnode(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            model.train()
            model.to(device)
            for i, batched_data in enumerate(training_loader):
                optimizer.zero_grad()
                x, edge_index = batched_data.x.to(device), batched_data.edge_index.to(device)
                y = batched_data.y.to(torch.float).to(device)
                out, infos, _ = model(x, edge_index)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                print(f"Run {run}/{args.runs} "
                      f"Epoch {epoch}/{args.epochs} "
                      f"Batch {i}/{batch_num}:"
                      f"loss {loss.item():.4f}")

            if epoch % args.eval_steps == 0:
                result = test(model, dataset, split_idx, evaluator)  # Modify
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()