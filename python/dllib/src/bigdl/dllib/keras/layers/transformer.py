#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
import math

from bigdl.nn.layer import Sum

from zoo.pipeline.api.keras.engine import ZooKerasLayer
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.models import Model
import zoo.pipeline.api.autograd as auto

if sys.version >= '3':
    long = int
    unicode = str


class TransformerLayer(ZooKerasLayer):
    """
    A self attention layer

    # Arguments
    nBlock: block number
    resid_drop: drop probability off projection
    attn_drop: drop probability of attention
    n_head: head number
    mask_attention: whether unidirectional or bidirectional
    embedding_layer: embedding layer
    """
    def __init__(self, n_block, resid_drop, attn_drop,
                 n_head, mask_attention, embedding_layer, input_shape, bigdl_type="float"):
        self.resid_drop = resid_drop
        self.attn_drop = attn_drop
        self.n_head = n_head
        self.mask_attention = mask_attention
        self.seq_len = input_shape[0]
        self.bigdl_type = bigdl_type
        if mask_attention:
            mask_value = np.tril(np.ones((self.seq_len, self.seq_len), dtype=bigdl_type))
            self.mask_value = auto.Constant(data=mask_value.reshape((1, 1,
                                                                     self.seq_len, self.seq_len)))

        input = Input(shape=list(input_shape))
        embedding = embedding_layer(input)
        hidden_size = embedding.get_output_shape()[-1]

        next_input = embedding

        for _ in range(n_block):
            output = self.block(next_input, hidden_size)
            next_input = output

        model = Model(input, next_input)
        self.value = model.value

    def block(self, x, size):
        g = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))
        g2 = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b2 = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))

        a = self.multi_head_self_attention(x, size)
        n = self.layer_norm(x + a, w=g, b=b)
        m = self.mlp(n, size)
        h = self.layer_norm(n + m, w=g2, b=b2)
        return h

    def multi_head_self_attention(self, x, size):
        c = Convolution1D(size * 3, 1, "normal", (0.0, 0.02))(x)
        query = c.slice(2, 0, size)
        key = c.slice(2, size, size)
        value = c.slice(2, size*2, size)
        q = self.split_heads(query)
        k = self.split_heads(key, k=True)
        v = self.split_heads(value)
        a = self.attn(q, k, v, True)
        m = self.merge_heads(a)
        n = Convolution1D(size, 1, "normal", (0.0, 0.02))(m)
        d = Dropout(self.resid_drop)(n)
        return d

    def split_heads(self, x, k=False):
        sizes = x.get_output_shape()[1:]
        shape = list(sizes + (sizes[-1]/self.n_head,))
        shape[-2] = self.n_head
        r = Reshape(shape)(x)
        if k:
            f = Permute((2, 3, 1))(r)
        else:
            f = Permute((2, 1, 3))(r)
        return f

    def merge_heads(self, x):
        p = auto.contiguous(Permute((2, 1, 3))(x))
        sizes = p.get_output_shape()[1:]
        merge_sizes = list((sizes[0], sizes[-1]*sizes[-2]))
        m = Reshape(merge_sizes)(p)
        return m

    def attn(self, q, k, v, scale=False):
        w = auto.mm(q, k)
        if scale:
            w = w / math.sqrt(v.get_output_shape()[-1])

        if self.mask_attention:
            w = w * self.mask_value + (self.mask_value * (-1.0) + 1.0) * (-1e9)

        w = Activation("softmax")(w)
        w = Dropout(self.attn_drop)(w)
        w = auto.mm(w, v)
        return w

    def layer_norm(self, x, w, b, e=1e-5):
        sizes = x.get_output_shape()[1:]
        u = auto.mean(x, len(sizes), True)
        s = auto.mean(auto.square(x - u), len(sizes), True)
        y = (x - u) / auto.sqrt(s + e)
        y = y * w + b
        return y

    def mlp(self, x, size):
        h = Convolution1D(size*4, 1, init="normal", limits=(0.0, 0.02))(x)
        a = self.gelu(h)
        h2 = Convolution1D(size, 1, init="normal", limits=(0.0, 0.02))(a)
        y = Dropout(self.resid_drop)(h2)
        return y

    def gelu(self, x):
        y = (auto.square(x) * x * 0.044715 + x) * (math.sqrt(2 / math.pi))
        y = Activation("tanh")(y) + 1.0
        y = x * 0.5 * y
        return y

    @classmethod
    def init_with_default_embedding(cls, vocab=40990, seq_len=77, n_block=12, resid_drop=0.1,
                                    attn_drop=0.1, n_head=12, hidden_size=768,
                                    embedding_drop=0.1, mask_attention=True):
        """
        vocab: vocabulary size of training data, default is 40990
        seq_len: max sequence length of training data, default is 77
        n_block: block number, default is 12
        resid_drop: drop probability of projection, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        n_head: head number, default is 12
        hidden_size: is also embedding size
        embedding_drop: drop probability of embedding layer, default is 0.1
        mask_attention: whether unidirectional or bidirectional, default is true(unidirectional)
        """
        from bigdl.nn.layer import Squeeze
        embedding = Sequential()

        embedding.add(Reshape([seq_len * 2], input_shape=(seq_len, 2)))\
            .add(Embedding(vocab, hidden_size, input_length=seq_len * 2))\
            .add(Dropout(embedding_drop))\
            .add(Reshape((seq_len, 2, hidden_size)))\
            .add(KerasLayerWrapper(Sum(dimension=3, squeeze=True)))
        # walk around for bug #1208, need remove this line after the bug fixed
        embedding.add(KerasLayerWrapper(Squeeze(dim=3)))

        return TransformerLayer(n_block, resid_drop, attn_drop, n_head, mask_attention,
                                embedding, input_shape=(seq_len, 2))
