import os
import math
import json

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import einsum, nn
from torch.utils.data import Dataset
import pytorch_lightning as pl


class MeanMaxPooling(nn.Module):
    """
    [B, S, E] -> [B, 2*E]
    """
    def __init__(self, axis=1, dropout=0.0):
        super().__init__()
        self.axis = axis
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, valid_length=None):
        """
        :param inputs: Tensor, shape [batch_size, seq_len, embedding_dim]
        :param valid_length: None or Tensor, valid len of token in the sequence with shape [batch_size]
        :return: Tensor, shape [batch_size, 2 * embedding_dim]
        """
        # TODO: broadcast indexing to mean over first vl
        mean_out = torch.mean(inputs, dim=self.axis) if valid_length is None \
            else torch.sum(inputs, dim=self.axis) / valid_length.add(1E-7).unsqueeze(1)
        max_out = torch.max(inputs, dim=self.axis).values
        outputs = torch.cat((mean_out, max_out), dim=1)
        outputs = self.dropout(outputs)
        return outputs


class Whitening:
    def __init__(self, vecs, n_components=248):
        super().__init__()
        self.vecs = vecs
        self.n_components = n_components
    
    def compute_kernel_bias(self, axis=0, keepdims=True):
        mu = self.vecs.mean(axis=axis, keepdims=keepdims)
        cov = np.cov(self.vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :self.n_components], -mu

    def transform_and_normalize(self, kernel=None, bias=None):
        if not (kernel is None or bias is None):
            vecs = (self.vecs + bias).dot(kernel)
        return vecs / (self.vecs**2).sum(axis=1, keepdims=True)**0.5
    

class SwiGLU(nn.Module):
    """
    Swish + GLU (SwiGLU) activation function used for Multilayer perceptron(MLP) intermediate activations in transformers.
    classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
    https://arxiv.org/abs/2002.05202
    """
    def forward(self, x):
        """
        takes in x input data and returns swiglu

        Args :
            x : input data
        Return : 
                SwiGLU applied activated output
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    

class LayerNorm(nn.Module):
    """
    Applies Layer Normalization for last certain number of dimensions.

    Args :
        dim : Dimension of input
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        """
        takes input data x and applies layer normalization

        Args :
            x : input data

        Return :
            The layer normalized values for the input data using gamma and beta init parameters.
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Residual(nn.Module):
    """
    Residual networks

    Args :
        fn : function
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, vl):
        """
        takes in x (input data) ,vl (values) and return residual values

        Args :
            x : input data
            vl : valid length to be used, Tensor, shape [batch_size]

        Return : 
            residual value after applying to a function
        """
        x_out= self.fn(x, vl)
        x_out += x
        return x_out

    
class RotaryEmbedding(nn.Module):
    """
    Rotatory positional (RoPE) embeddings, paper -  https://arxiv.org/abs/2104.09864.
    RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation

    Args :
        dim : dimensions
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        """
        takes in max_seq_len, *, device as input and return embeddings

        Args :
            max_seq_len : input data
            * :
            device : device to be used, cpu or gpu

        Return : 
            embeddings
        """
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
    

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. 
    The positional encodings have the same dimension as the embeddings so that the two can be summed

    Args :
        d_model: dimension of token embedding
        max_len: max length of sequence
        dropout: dropout field for regularization
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Takes in the input value and add the positional value, apply the dropout on top of it and then return the final value.

        Args :
            x: input data

        Return :
            output of positional encoding

        Shape :
            x: [batch_size, seq_len, embedding_dim]
            
            out : [batch_size, seq_len, embedding_dim]

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class VZArtifactCallback(pl.callbacks.Callback):
    def __init__(self, torch_model_params, model_dir, params_to_pop=None):
        self.model_dir = model_dir
        self.torch_model_params = torch_model_params
        if params_to_pop:
            for p in params_to_pop:
                self.torch_model_params.pop(p, None)

    def on_train_start(self, trainer, pl_module):
        torch_model_params_path = os.path.join(self.model_dir, "torch_model_params.json")
        with open(torch_model_params_path, "w") as f:
            json.dump(self.torch_model_params, f)


    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            model_path = os.path.join(self.model_dir, f"model_{trainer.current_epoch}.pt")
            torch.save(pl_module.torch_model.state_dict(), model_path)


class ViewDataset2(Dataset):
    """
    If from_hdfs is True, will read data from hdfs. hdfs_data_dir and hdfs_lookup_dirs are required.
    Otherwise will read local data. local_data_data_dir and local_lookup_dirs are required.
    """

    def __init__(self, deep_cols, pn_seq_col, ic_seq_col, vl_col, search_col, label_col, task_type_cols, current_col,
                 current_meta_col, page_meta_cols=None, page_meta_wide_cols=None, item_meta_cols=None,
                 item_meta_wide_cols=None, wide_cols=None,
                 shared_cols=None, transform=None, tokenizer=None, pd_df=None, from_hdfs=False,
                 local_data_dir=None, local_lookup_dirs=None, hdfs_data_dir=None,
                 hdfs_lookup_dirs=None, parts_num=9999, data_file_format="parquet", lookup_file_format="json"):
        self.transform = transform
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_cols = deep_cols
        self.wide_cols = wide_cols
        self.task_type_cols = task_type_cols
        self.current_col = current_col
        self.current_meta_col = current_meta_col
        self.pn_seq_col = pn_seq_col
        self.ic_seq_col = ic_seq_col
        self.item_meta_cols = item_meta_cols
        self.page_meta_cols = page_meta_cols
        self.item_meta_wide_cols = item_meta_wide_cols
        self.page_meta_wide_cols = page_meta_wide_cols
        self.shared_cols = shared_cols
        self.vl_col = vl_col
        self.search_col = search_col
        self.label_col = label_col
        self.local_data_dir = local_data_dir
        self.local_lookup_dirs = local_lookup_dirs
        self.from_hdfs = from_hdfs
        self.hdfs_data_dir = hdfs_data_dir
        self.hdfs_lookup_dirs = hdfs_lookup_dirs
        self.parts_num = parts_num
        self.data_file_format = data_file_format
        self.lookup_file_format = lookup_file_format
        # load data
        self.deep_np = []
        self.wide_np = []
        self.task_type_np = []
        self.current_np = []
        self.current_meta_np = []
        self.pn_seq_np = []
        self.ic_seq_np = []
        self.item_meta_np = []
        self.page_meta_np = []
        self.item_meta_wide_np = []
        self.page_meta_wide_np = []
        self.shared_np = []
        self.vl_np = []
        self.search_np = []
        self.label_np = []
        self.lookups = {}

        assert local_data_dir, local_lookup_dirs
        if pd_df is not None:
            self._append_np_from_df(pd_df)
        else:
            self._load_lookups_from_local()
            self._load_data_from_local()

        # convert dataframe to numpy
        self.deep_np = np.concatenate(self.deep_np, axis=0)
        if self.wide_cols is not None:
            self.wide_np = np.concatenate(self.wide_np, axis=0)
        self.pn_seq_np = np.concatenate(self.pn_seq_np, axis=0)
        self.ic_seq_np = np.concatenate(self.ic_seq_np, axis=0)
        if self.item_meta_cols is not None:
            self.item_meta_np = np.concatenate(self.item_meta_np, axis=0)
        if self.page_meta_cols is not None:
            self.page_meta_np = np.concatenate(self.page_meta_np, axis=0)
        if self.item_meta_wide_cols is not None:
            self.item_meta_wide_np = np.concatenate(self.item_meta_wide_np, axis=0)
        if self.page_meta_wide_cols is not None:
            self.page_meta_wide_np = np.concatenate(self.page_meta_wide_np, axis=0)
        if self.shared_cols is not None:
            self.shared_np = np.concatenate(self.shared_np, axis=0)
        self.vl_np = np.concatenate(self.vl_np, axis=0)
        self.task_type_np = np.concatenate(self.task_type_np, axis=0)
        self.current_np = np.concatenate(self.current_np, axis=0)
        self.current_meta_np = np.concatenate(self.current_meta_np, axis=0)
        self.search_np = np.concatenate(self.search_np, axis=0)
        self.label_np = np.concatenate(self.label_np, axis=0)
        self.len_data = len(self.deep_np)

    @staticmethod
    def _read_df(path, file_format):
        if file_format == "json":
            return pd.read_json(path, lines=True)
        elif file_format == "csv":
            return pd.read_csv(path, engine="pyarrow")
        elif file_format == "parquet":
            return pd.read_parquet(path, engine="pyarrow")

    def _append_np_from_df(self, df):
        self.deep_np.append(np.array(df[self.deep_cols]))
        self.wide_np.append(np.array(df[self.wide_cols]))
        self.pn_seq_np.append(np.array(df[self.pn_seq_col].to_list()))
        self.ic_seq_np.append(np.array(df[self.ic_seq_col].to_list()))
        if self.item_meta_cols is not None:
            self.item_meta_np.append(np.array(df[self.item_meta_cols].to_list()))
        if self.page_meta_cols is not None:
            self.page_meta_np.append(np.array(df[self.page_meta_cols].to_list()))
        if self.item_meta_wide_cols is not None:
            self.item_meta_wide_np.append(np.array(df[self.item_meta_wide_cols].to_list()))
        if self.page_meta_wide_cols is not None:
            self.page_meta_wide_np.append(np.array(df[self.page_meta_wide_cols].values.tolist()))
        if self.shared_cols is not None:
            self.shared_np.append(np.array(df[self.shared_cols]))
        self.vl_np.append(np.array(df[self.vl_col]))
        self.task_type_np.append(np.array(df[self.task_type_cols]))
        self.current_np.append(np.array(df[self.current_col]))
        self.current_meta_np.append(np.array(df[self.current_meta_col]))
        self.search_np.append(np.array(df[self.search_col]))
        self.label_np.append(np.array(df[self.label_col].astype(float)))

    def _load_data_from_local(self):
        f_paths = [self.local_data_dir + f_name for f_name in os.listdir(self.local_data_dir) if
                   f_name.endswith(self.data_file_format)]
        for i, f_path in enumerate(f_paths):
            df = self._read_df(f_path, self.data_file_format)
            self._append_np_from_df(df)
            if i > self.parts_num:
                break

    def _load_lookups_from_local(self):
        for local_dir in self.local_lookup_dirs:
            lookup_name = os.path.split(local_dir[:-1])[1] if local_dir.endswith("/") else os.path.split(local_dir)[1]
            local_paths = [
                local_dir + file_name for file_name in os.listdir(local_dir)
                if file_name.endswith(self.lookup_file_format)
            ]
            assert len(local_paths) == 1, f"should only have one lookup table per folder, now have {len(local_paths)}"
            df = self._read_df(local_paths[0], file_format=self.lookup_file_format)
            self.lookups[lookup_name] = df

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        deep = [i for i in self.deep_np[idx]]
        wide = [int(i) for i in self.wide_np[idx]]
        pn_seq = torch.tensor(self.pn_seq_np[idx])
        ic_seq = torch.tensor(self.ic_seq_np[idx])
        item_meta_in = torch.tensor(self.item_meta_np[idx])
        page_meta_wide_in = torch.tensor(self.page_meta_wide_np[idx])
        vl = self.vl_np[idx]
        task_type = [int(i) for i in self.task_type_np[idx]]
        current = self.current_np[idx]
        current_meta = self.current_meta_np[idx]
        search = self.search_np[idx]
        label = self.label_np[idx]
        data = (
        deep, pn_seq, ic_seq, vl, task_type, current, current_meta, wide, search, item_meta_in, page_meta_wide_in)
        return data, label

    def get_feature_num(self, col):
        return int(self.lookups[col[:-4]][col].max()) + 1

    def export_lookups(self, local_dir, file_format):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        for lookup_name, lookup_df in self.lookups.items():
            output_path = os.path.join(local_dir, f"{lookup_name}.{file_format}")
            if file_format == "json":
                lookup_df.to_json(output_path)
            elif file_format == "csv":
                lookup_df.to_csv(output_path, index=False)
            else:
                raise NotImplementedError()
