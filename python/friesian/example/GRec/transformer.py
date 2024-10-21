from einops import rearrange

from utils import *

#creating rotary positional embedding for giving a sequence to the data
def rotate_half(x):
    """
    The `rotate_half` method takes in a tensor `x` and rotates its last dimension by 180 degrees. The tensor is first rearranged to have a new dimension `j` which is equal to 2. The tensor is then split into two tensors `x1` and `x2` along the second to last dimension. The two tensors are then concatenated along the last dimension after the second tensor is negated. The resulting tensor has the same shape as the input tensor.
    """

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    """
    The `apply_rotary_pos_emb` method takes in two tensors `pos` and `t`. The `pos` tensor is a positional encoding tensor and `t` is the input tensor. The method applies a rotary positional encoding to the input tensor `t` using the positional encoding tensor `pos`. The method first applies a cosine function to the positional encoding tensor `pos` and element-wise multiplies it with the input tensor `t`. The method then applies a sine function to the positional encoding tensor `pos`, rotates it by 180 degrees using the `rotate_half` method, and element-wise multiplies it with the input tensor `t`. The two resulting tensors are then added together to produce the final output tensor. 
    """

    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class ParallelTransformerBlock(nn.Module):
    """
    ## Class - ParallelTransformerBlock
    ## Class Comment - This class implements a PyTorch implementation of the ParallelTransformerBlock Transformer model. It includes methods for creating key padding masks, getting rotary embeddings, and forward propagation through the model.
    
    Instantiates Rotary Embedding ( RoPE https://arxiv.org/abs/2104.09864), Linear and SwiGLU Activation Layer
    (a variant of GLU https://arxiv.org/pdf/2002.05202.pdf)
    The architecture is based on the paper "PaLM: Scaling Language Modeling with Pathways" (https://arxiv.org/abs/2204.02311)
    The parallel formulation of transformer results in faster training speed at large scales

    
    ## Class Example - 

    import torch
    from vz_recommender.models.transformer import  ParallelTransformerBlock

    model = ParallelTransformerBlock(dim=512, dim_head=64, heads=8)
    input_data = torch.randn(10, 20, 512)
    output = model(input_data)
    """

    def __init__(self, dim, dim_head, heads, ff_mult=4, moe_kwargs=None):
        """
        ## Method - __init__()
        ## Method Comment - Initializes the ParallelTransformerBlock model with the given parameters. Sets up the necessary layers and buffers for forward propagation.
        ## Method Arguments - 
        - dim (int): The dimension of the input data.
        - dim_head (int): The dimension of the attention heads.
        - heads (int): The number of attention heads.
        - ff_mult (int): The multiplier for the feedforward layer.
        - moe_kwargs (dict): Optional dictionary of arguments for the mixture of experts layer.
        ## Method Return - None
        ## Method Shape - O(1)
        """
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        
        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)
        
    @staticmethod
    def create_key_padding_mask(seq_in, valid_length=None):
        """
        ## Method - create_key_padding_mask()
        ## Method Comment - Creates a key padding mask for the input sequence. Used to mask out padding tokens during attention calculations.
        ## Method Arguments - 
        - seq_in (torch.Tensor): The input sequence tensor.
        - valid_length (torch.Tensor): Optional tensor of valid lengths for each sequence in the batch.
        ## Method Return - A boolean mask tensor of shape (batch_size, seq_length).
        ## Method Shape - O(seq_length)
        """
        device = seq_in.device
        vl_len = torch.cat((seq_in.size(0)*[torch.tensor([seq_in.size(1)])]), dim=0).to(device) if valid_length is None else valid_length
        mask = torch.arange(seq_in.size(1)).repeat(seq_in.size(0), 1).to(device)
        mask = ~mask.lt(vl_len.unsqueeze(1))
        return mask

    def get_mask(self, n, device):
        """
        ## Method - get_mask()
        ## Method Comment - Gets a triangular mask for the attention calculation. Used to prevent attention from attending to future tokens.
        ## Method Arguments - 
        - n (int): The length of the sequence.
        - device (torch.device): The device to create the mask on.
        ## Method Return - A boolean mask tensor of shape (seq_length, seq_length).
        ## Method Shape - O(seq_length^2)
        """
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        """
        ## Method - get_rotary_embedding()
        ## Method Comment - Gets the rotary positional embedding for the input sequence.
        ## Method Arguments - 
        - n (int): The length of the sequence.
        - device (torch.device): The device to create the embedding on.
        ## Method Return - A tensor of shape (seq_length, dim_head).
        ## Method Shape - O(seq_length * dim_head)
        """

        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, vl=None):
        """
        ## Method - forward()
        ## Method Comment - Performs forward propagation through the ParallelTransformerBlock model.
        ## Method Arguments - 
        - x (torch.Tensor): The input sequence tensor of shape (batch_size, seq_length, dim).
        - vl (torch.Tensor): Optional tensor of valid lengths for each sequence in the batch.
        ## Method Return - The output tensor of shape (batch_size, seq_length, dim).
        ## Method Shape - O(seq_length^2 * dim)
        """

        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        # attention queries, keys, values, and feedforward inner
        #creates attention heads 
        #shape of q : [batch size, 2*vl]
        #shape of k : [batch_size, 2*vl]
        #shape of ff : [batch_size, 8*vl]
        x = self.fused_attn_ff_proj(x)
        q, k, v, ff = x.split(self.fused_dims, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        mask = self.create_key_padding_mask(seq_in=x, valid_length=vl)
        sim = sim.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.attn_out(out) + self.ff_out(ff)
        return out
    

class ParallelTransformerAEP(nn.Module):
    """
    This class implements a Parallel Transformer Autoencoder with Page-Item Attention for recommendation systems. It takes in page and item embeddings, along with various meta data, and applies a parallel transformer block to encode the data and generate a recommendation.
        
    ## Class Example - 
    import torch 
    from torch import nn
    from vz_recommender.models.transformer import ParallelTransformerAEP

    model = ParallelTransformerAEP(page_embedding, item_embedding, dim=512, dim_head=64, heads=8, num_layers=6, num_page_meta_wide=0, page_meta_wide_embed_dim=0, num_item_meta_wide=0, item_meta_wide_embed_dim=0, ff_mult=4, seq_pooling_dropout=0.0, page_meta_embedding=None, item_meta_embedding=None, item_pre_embedding=None, moe_kwargs=None)
    output = model(page_in, item_in, item_meta_in, vl_in, page_meta_in=None, page_meta_wide_in=None, item_meta_wide_in=None)
    """
 
    def __init__(self, page_embedding, item_embedding, dim, dim_head, heads, num_layers, num_page_meta_wide=0, page_meta_wide_embed_dim=0, num_item_meta_wide=0, item_meta_wide_embed_dim=0, ff_mult=4, seq_pooling_dropout=0.0, page_meta_embedding=None, item_meta_embedding=None, item_pre_embedding=None, moe_kwargs=None):
        """
        ## Method - __init__()
        ## Method Comment - Initializes the ParallelTransformerAEP class with the given parameters.
        ## Method Arguments - 
        - page_embedding: torch.nn.Embedding - Embedding for pages
        - item_embedding: torch.nn.Embedding - Embedding for items
        - dim: int - Dimension of the model
        - dim_head: int - Dimension of the attention head
        - heads: int - Number of attention heads
        - num_layers: int - Number of transformer layers
        - num_page_meta_wide: int - Number of wide meta data for pages
        - page_meta_wide_embed_dim: int - Embedding dimension for wide meta data for pages
        - num_item_meta_wide: int - Number of wide meta data for items
        - item_meta_wide_embed_dim: int - Embedding dimension for wide meta data for items
        - ff_mult: int - Multiplier for feedforward network
        - seq_pooling_dropout: float - Dropout rate for sequence pooling
        - page_meta_embedding: torch.nn.Embedding - Embedding for page meta data
        - item_meta_embedding: torch.nn.Embedding - Embedding for item meta data
        - item_pre_embedding: torch.nn.Embedding - Embedding for item pre data
        - moe_kwargs: dict - Dictionary of arguments for mixture of experts
        ## Method Return - None
        ## Method Shape - N/A
        """
        super().__init__()
        self.page_embedding = page_embedding
        self.page_meta_embedding = page_meta_embedding
        if num_page_meta_wide > 0:
            self.num_page_meta_wide = num_page_meta_wide
            self.page_meta_wide_dense = nn.Linear(num_page_meta_wide, page_meta_wide_embed_dim)
            self.page_meta_wide_act = nn.LeakyReLU(0.2)
        if num_page_meta_wide > 1:
            self.page_meta_wide_batch_norm = nn.BatchNorm1d(num_page_meta_wide)
        self.item_embedding = item_embedding
        self.item_meta_embedding = item_meta_embedding
        self.item_pre_embedding = item_pre_embedding
        if num_item_meta_wide > 0:
            self.num_item_meta_wide = num_item_meta_wide
            self.item_meta_wide_dense = nn.Linear(num_item_meta_wide, item_meta_wide_embed_dim)
            self.item_meta_wide_act = nn.LeakyReLU(0.2)
        if num_item_meta_wide > 1:
            self.item_meta_wide_batch_norm = nn.BatchNorm1d(num_item_meta_wide)
        self.seq_pooling_dp = MeanMaxPooling(dropout=seq_pooling_dropout)
        self.seq_dense = torch.nn.Linear(2 * dim, dim)  
        self.num_layers = num_layers
        
        self.ptransformer = nn.ModuleList([
            Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, moe_kwargs=moe_kwargs))
            for _ in range(self.num_layers)
        ])
        
    def forward(self, page_in, item_in, item_meta_in, vl_in, page_meta_in=None, page_meta_wide_in=None, item_meta_wide_in=None):
        """
        ## Method - forward()
        ## Method Comment - Applies the forward pass of the ParallelTransformerAEP model to the given input data.
        ## Method Arguments - 
        - page_in: torch.Tensor - Input tensor for pages
        - item_in: torch.Tensor - Input tensor for items
        - item_meta_in: torch.Tensor - Input tensor for item meta data
        - vl_in: torch.Tensor - Input tensor for visual and language data
        - page_meta_in: torch.Tensor - Input tensor for page meta data
        - page_meta_wide_in: List[torch.Tensor] - List of input tensors for wide page meta data
        - item_meta_wide_in: List[torch.Tensor] - List of input tensors for wide item meta data
        ## Method Return - torch.Tensor - Output tensor of the ParallelTransformerAEP model
        ## Method Shape - (batch_size, dim)
        """
        page_embed_out = self.page_embedding(page_in.long())
        item_embed_out = self.item_embedding(item_in.long())
        
        if page_meta_in is not None:
            page_meta_embed_out = self.page_meta_embedding(page_meta_in.long()) 
        if item_meta_in is not None:
            item_meta_embed_out = self.item_meta_embedding(item_meta_in.long()) 
            item_pre_embed_out = self.item_pre_embedding(item_in.long())
        
        if page_meta_wide_in is not None:
            page_meta_wide_in_list = [wide_i.float() for wide_i in page_meta_wide_in]
            page_meta_wide_cat = torch.stack(page_meta_wide_in_list, dim=0)
            if self.num_page_meta_wide > 1:
                page_meta_wide_out_norm = self.page_meta_wide_batch_norm(page_meta_wide_cat) 
            else:
                page_meta_wide_out_norm = page_meta_wide_cat
            page_meta_wide_out_norm = torch.permute(page_meta_wide_out_norm, (0,2,1))
            page_meta_wide_out_norm = self.page_meta_wide_dense(page_meta_wide_out_norm)
            page_meta_wide_out_norm = self.page_meta_wide_act(page_meta_wide_out_norm)
            if page_meta_in is not None:
                page_full_out = torch.cat((page_embed_out, page_meta_embed_out, page_meta_wide_out_norm), 2)
            else:
                page_full_out = torch.cat((page_embed_out, page_meta_wide_out_norm), 2)
        else:
            if page_meta_in is not None:
                page_full_out = torch.cat((page_embed_out, page_meta_embed_out), 2)
            else:
                page_full_out = page_embed_out
            
        if item_meta_wide_in is not None:
            item_meta_wide_in_list = [wide_i.float() for wide_i in item_meta_wide_in]
            item_meta_wide_cat = torch.stack(item_meta_wide_in_list, dim=0)
            if self.num_item_meta_wide > 1:
                item_meta_wide_out_norm = self.item_meta_wide_batch_norm(item_meta_wide_cat) 
            else:
                item_meta_wide_out_norm = item_meta_wide_cat
            item_meta_wide_out_norm = torch.permute(item_meta_wide_out_norm, (0,2,1))
            item_meta_wide_out_norm = self.item_meta_wide_dense(item_meta_wide_out_norm)
            item_meta_wide_out_norm = self.item_meta_wide_act(item_meta_wide_out_norm)
            if item_meta_in is not None:
                item_full_out = torch.cat((item_embed_out, item_meta_embed_out, item_pre_embed_out, item_meta_wide_out_norm), 2)
            else:
                item_full_out = torch.cat((item_embed_out, item_meta_wide_out_norm), 2)
        else:
            if item_meta_in is not None:
                item_full_out = torch.cat((item_embed_out, item_meta_embed_out, item_pre_embed_out), 2)
            else: 
                item_full_out = item_embed_out
          
        x = torch.mul(page_full_out, item_full_out)
        for i in range(self.num_layers):
            x = self.ptransformer[i](x, vl_in)

        out = self.seq_pooling_dp(x)
        out = self.seq_dense(out)        
        return out
