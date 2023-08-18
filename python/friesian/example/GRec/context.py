from typing import *

import torch
from torch import Tensor, nn


class ContextHead(nn.Module):
    """
    Transformer using deep and wide context features along with embeddings shared with sequence transformer
  
    Args:
        deep_dims: size of the dictionary of embeddings
        item_embedding: item embedding shared with sequence transformer.
        deep_embed_dims: the size of each embedding vector, can be either int or list of int
        num_wide: the number of wide input features (default=0)
        wad_embed_dim: no. of output features (default=64)
        num_shared: the number of embedding shared with sequence transformer (default=1)
        shared_embeddings_weight: list of embedding shared with candidate generation model
    """
    def __init__(self, deep_dims, item_embedding=None, deep_embed_dims=100, num_wide=0, wad_embed_dim=64,
                 num_shared=0, shared_embeddings_weight=None):
        super().__init__()
        self.num_wide = num_wide
        if isinstance(deep_embed_dims, int):
            if shared_embeddings_weight is None:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(deep_dim, deep_embed_dims)
                    for deep_dim in deep_dims
                ])
            else:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(deep_dim, deep_embed_dims)
                    for deep_dim in deep_dims[:-len(shared_embeddings_weight)]
                ])
                from_pretrained = nn.ModuleList([
                    nn.Embedding.from_pretrained(shared_embedding_weight, freeze=True)
                    for shared_embedding_weight in shared_embeddings_weight
                ])
                self.deep_embedding.extend(from_pretrained)
        elif isinstance(deep_embed_dims, list) or isinstance(deep_embed_dims, tuple):
            self.deep_embedding = nn.ModuleList([
                nn.Embedding(deep_dim, deep_embed_dim)
                for deep_dim, deep_embed_dim in zip(deep_dims, deep_embed_dims)
            ])
        else:
            raise NotImplementedError()

        self.ctx_pe = False

        if self.num_wide > 1:
            self.wide_batch_norm = nn.BatchNorm1d(num_wide)
        self.deep_embed_dims = deep_embed_dims
        if item_embedding and num_shared:
            self.shared_embed = nn.ModuleList([
                item_embedding
                for _ in range(num_shared)
            ])
        else:
            self.shared_embed = None
        self.deep_dense = nn.Linear((len(deep_dims) + num_shared) * deep_embed_dims, wad_embed_dim//2)
        self.deep_act = nn.LeakyReLU(0.2)
        if self.num_wide:
            self.wide_dense = nn.Linear(num_wide, wad_embed_dim//2)
            self.wide_act = nn.LeakyReLU(0.2)

    def forward(self, deep_in: List[Tensor], wide_in: List[Tensor] = None, shared_in: List[Tensor] = None):
        """
        Input is deep, wide & shared embedding
        
        Args:
            deep_in: list
            wide_in: list
            shared_in: list (default=None).

        Return:
            ctx_out: Tensor
        
        Shape:
            deep_in: [batch_size, deep_dims]
            wide_in: [batch_size, num_wide]
            shared_in: [batch_size, num_shared]
            ctx_out: [batch_size, len(deep_dims)*deep_embed_dims+(num_shared*seq_embed_dim)+num_wide]
        """
        deep_embedding_list = [self.deep_embedding[i](input_deep).unsqueeze(1)
                               for i, input_deep in enumerate(deep_in)]
        if shared_in is not None and self.shared_embed is not None:
            shared_in_list = [self.shared_embed[i](input_shared).unsqueeze(1)
                              for i, input_shared in enumerate(shared_in)]
            deep_embedding_list.extend(shared_in_list)

        deep_out = torch.cat(deep_embedding_list, dim=2).squeeze(1)
        deep_out = self.deep_dense(deep_out)
        deep_out = self.deep_act(deep_out)
        if self.num_wide:
            wide_in_list = [wide_i.float() for wide_i in wide_in]
            wide_cat = torch.stack(wide_in_list, dim=0)
            wide_out = torch.transpose(wide_cat, dim1=1, dim0=0)
            if self.num_wide != 1:
                wide_out_norm = self.wide_batch_norm(wide_out)
            else:
                wide_out_norm = wide_out
            wide_out_norm = self.wide_dense(wide_out_norm)
            wide_out_norm = self.wide_act(wide_out_norm)
            ctx_out = torch.cat((deep_out, wide_out_norm), dim=1)
            return ctx_out
        else:
            return deep_out
