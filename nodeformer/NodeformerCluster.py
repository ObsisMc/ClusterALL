import torch
import torch.nn as nn
from AbstractClusteror import AbstractClusteror, AbstractClusterDataset, AbstractClusterLoader
from dataset import NCDataset


class NodeformerCluster(AbstractClusteror):
    def __init__(self, encoder: nn.Module, in_channels, hidden_channels, out_channels, decode_channels, num_parts,
                 **kwargs):
        super().__init__(encoder, in_channels, hidden_channels, out_channels, decode_channels, num_parts, **kwargs)

    def encode_forward(self, x, edge_index, **kwargs) -> (torch.Tensor, dict):
        adjs, tau, edge_mask = kwargs["adjs"], kwargs.get("tau", 0.25), kwargs.get("edge_mask")
        x, loss = self.encoder(x, adjs, tau=tau, edge_mask=edge_mask)
        return x, {"loss": loss}


class NodeformerClusterNCDataset(NCDataset, AbstractClusterDataset):
    def __init__(self, name, dataset, data, split_idx: dict, num_parts: int, load_path=None):
        super().__init__(name)  # init the first father
        AbstractClusterDataset.__init__(self, dataset, data, split_idx, num_parts, load_path)


class NodeformerClusterLoader(AbstractClusterLoader):
    def __init__(self, dataset: AbstractClusterDataset, split_name: str, is_eval: bool, batch_size: int, shuffle):
        super().__init__(dataset, split_name, is_eval, batch_size, shuffle)
