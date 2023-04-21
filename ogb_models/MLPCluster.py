import torch
import torch.nn as nn
from AbstractClusteror import AbstractClusteror, AbstractClusterDataset, AbstractClusterLoader


class MLPCluster(AbstractClusteror):
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, decode_channels, num_parts,
                 **kwargs):
        super().__init__(encoder, in_channels, hidden_channels, out_channels, decode_channels, num_parts, **kwargs)

    def encode_forward(self, x, edge_index, **kwargs) -> (torch.Tensor, dict):
        x = self.encoder(x)
        return x, None


class MLPClusterDataset(AbstractClusterDataset):
    def __init__(self, dataset, data, split_idx: dict, num_parts: int, load_path: str = None):
        super().__init__(dataset, data, split_idx, num_parts, load_path)


class MLPClusterLoader(AbstractClusterLoader):
    def __init__(self, dataset: AbstractClusterDataset, split_name: str, is_eval: bool, batch_size: int, shuffle):
        super().__init__(dataset, split_name, is_eval, batch_size, shuffle)
