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
from bigdl.nn.layer import Layer
from zoo.common.utils import callZooFunc

from zoo.models.common import ZooModel
from zoo.pipeline.api.keras.engine import ZooKerasLayer
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.models import Model
import zoo.pipeline.api.autograd as auto

if sys.version >= '3':
    long = int
    unicode = str


def layer_norm(x, w, b, e=1e-5):
    sizes = x.get_output_shape()[1:]
    u = auto.mean(x, len(sizes), True)
    s = auto.mean(auto.square(x - u), len(sizes), True)
    y = (x - u) / auto.sqrt(s + e)
    y = y * w + b
    return y


class TransformerLayer(ZooKerasLayer):
    """
    A self attention layer

    Input is a list which consists of 2 ndarrays.
    1. Token id ndarray: shape [batch, seqLen] with the word token indices in the vocabulary
    2. Position id ndarray: shape [batch, seqLen] with positions in the sentence.
    Output is a list which contains:
    1. The states of Transformer layer.
    2. The pooled output which processes the hidden state of the last layer with regard to the first
    token of the sequence. This would be useful for segment-level tasks.

    # Arguments
    nBlock: block number
    hidden_drop: drop probability off projection
    attn_drop: drop probability of attention
    n_head: head number
    initializer_range: weight initialization range
    bidirectional: whether unidirectional or bidirectional
    output_all_block: whether output all blocks' output
    embedding_layer: embedding layer
    input_shape: input shape
    """

    def __init__(self, n_block, hidden_drop, attn_drop, n_head, initializer_range, bidirectional,
                 output_all_block, embedding_layer, input_shape, intermediate_size=0,
                 bigdl_type="float"):
        self.hidden_drop = hidden_drop
        self.attn_drop = attn_drop
        self.n_head = n_head
        self.initializer_range = initializer_range
        self.output_all_block = output_all_block
        self.bidirectional = bidirectional
        self.intermediate_size = intermediate_size
        self.seq_len = input_shape[0][0]
        self.bigdl_type = bigdl_type
        if not bidirectional:
            mask_value = np.tril(np.ones((self.seq_len, self.seq_len), dtype=bigdl_type))
            self.mask_value = auto.Constant(data=mask_value.reshape((1, 1,
                                                                     self.seq_len, self.seq_len)))

        (extended_attention_mask, embedding_inputs, inputs) = self.build_input(input_shape)
        embedding = embedding_layer(embedding_inputs)
        hidden_size = embedding.get_output_shape()[-1]

        next_input = embedding

        output = [None] * n_block
        output[0] = self.block(next_input, hidden_size, extended_attention_mask)

        for index in range(n_block - 1):
            o = self.block(output[index], hidden_size, extended_attention_mask)
            output[index + 1] = o

        pooler_output = self.pooler(output[-1], hidden_size)
        model = Model(inputs, output.append(pooler_output)) if output_all_block \
            else Model(inputs, [output[-1], pooler_output])
        self.value = model.value

    def build_input(self, input_shape):
        if any(not isinstance(i, tuple) and not isinstance(i, list) for i in input_shape):
            raise TypeError('TransformerLayer input must be a list of ndarray (consisting'
                            ' of input sequence, sequence positions, etc.)')

        inputs = [Input(list(shape)) for shape in input_shape]
        return None, inputs, inputs

    def block(self, x, size, attention_mask=None, eplision=1e-5):
        g = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))
        g2 = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b2 = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))

        a = self.multi_head_self_attention(x, size, attention_mask)
        n = layer_norm(x + a, w=g, b=b, e=eplision)
        m = self.mlp(n, size)
        h = layer_norm(n + m, w=g2, b=b2, e=eplision)
        return h

    def projection_layer(self, output_size):
        return Convolution1D(output_size, 1, "normal", (0.0, self.initializer_range))

    def multi_head_self_attention(self, x, size, attention_mask=None):
        c = self.projection_layer(size * 3)(x)
        query = c.slice(2, 0, size)
        key = c.slice(2, size, size)
        value = c.slice(2, size * 2, size)
        q = self.split_heads(query, self.n_head)
        k = self.split_heads(key, self.n_head, k=True)
        v = self.split_heads(value, self.n_head)
        a = self.attn(q, k, v, True, attention_mask)
        m = self.merge_heads(a)
        n = self.projection_layer(size)(m)
        d = Dropout(self.hidden_drop)(n)
        return d

    def attn(self, q, k, v, scale=False, attention_mask=None):
        w = auto.mm(q, k)
        if scale:
            w = w / math.sqrt(v.get_output_shape()[-1])

        if not self.bidirectional:
            w = w * self.mask_value + (self.mask_value * (-1.0) + 1.0) * (-1e9)
        if attention_mask:
            w = w + attention_mask

        w = Activation("softmax")(w)
        w = Dropout(self.attn_drop)(w)
        w = auto.mm(w, v)
        return w

    def mlp(self, x, hidden_size):
        size = self.intermediate_size if self.intermediate_size > 0 else hidden_size * 4
        h = self.projection_layer(size)(x)
        a = self.gelu(h)
        h2 = self.projection_layer(hidden_size)(a)
        y = Dropout(self.hidden_drop)(h2)
        return y

    def gelu(self, x):
        y = (auto.square(x) * x * 0.044715 + x) * (math.sqrt(2 / math.pi))
        y = Activation("tanh")(y) + 1.0
        y = x * 0.5 * y
        return y

    def split_heads(self, x, n_head, k=False):
        sizes = x.get_output_shape()[1:]
        shape = list(sizes + (int(sizes[-1] / n_head),))
        shape[-2] = n_head
        r = Reshape(shape)(x)
        if k:
            f = Permute((2, 3, 1))(r)
        else:
            f = Permute((2, 1, 3))(r)
        return f

    def merge_heads(self, x):
        p = auto.contiguous(Permute((2, 1, 3))(x))
        sizes = p.get_output_shape()[1:]
        merge_sizes = list(sizes[:-2] + (sizes[-1] * sizes[-2],))
        m = Reshape(merge_sizes)(p)
        return m

    def pooler(self, x, hidden_size):
        first_token = Select(1, 0)(x)
        pooler_output = Dense(hidden_size)(first_token)
        o = Activation("tanh")(pooler_output)
        return o

    @classmethod
    def init(cls, vocab=40990, seq_len=77, n_block=12, hidden_drop=0.1,
             attn_drop=0.1, n_head=12, hidden_size=768,
             embedding_drop=0.1, initializer_range=0.02,
             bidirectional=False, output_all_block=False):
        """
        vocab: vocabulary size of training data, default is 40990
        seq_len: max sequence length of training data, default is 77
        n_block: block number, default is 12
        hidden_drop: drop probability of projection, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        n_head: head number, default is 12
        hidden_size: is also embedding size
        embedding_drop: drop probability of embedding layer, default is 0.1
        initializer_range: weight initialization range, default is 0.02
        bidirectional: whether unidirectional or bidirectional, default is unidirectional
        output_all_block: whether output all blocks' output
        """
        if hidden_size < 0:
            raise TypeError('hidden_size must be greater than 0 with default embedding layer')
        from bigdl.nn.layer import Squeeze
        word_input = InputLayer(input_shape=(seq_len,))
        postion_input = InputLayer(input_shape=(seq_len,))

        embedding = Sequential()
        embedding.add(Merge(layers=[word_input, postion_input], mode='concat')) \
            .add(Reshape([seq_len * 2])) \
            .add(Embedding(vocab, hidden_size, input_length=seq_len * 2,
                           weights=np.random.normal(0.0, initializer_range, (vocab, hidden_size))))\
            .add(Dropout(embedding_drop)) \
            .add(Reshape((seq_len, 2, hidden_size))) \
            .add(KerasLayerWrapper(Sum(dimension=3, squeeze=True)))
        # walk around for bug #1208, need remove this line after the bug fixed
        embedding.add(KerasLayerWrapper(Squeeze(dim=3)))

        shape = ((seq_len,), (seq_len,))
        return TransformerLayer(n_block, hidden_drop, attn_drop, n_head, initializer_range,
                                bidirectional, output_all_block, embedding, input_shape=shape)


class BERT(TransformerLayer):
    """
    A self attention layer.
    Input is a list which consists of 4 ndarrays.
    1. Token id ndarray: shape [batch, seqLen] with the word token indices in the vocabulary
    2. Token type id ndarray: shape [batch, seqLen] with the token types in [0, 1].
       0 means `sentence A` and 1 means a `sentence B` (see BERT paper for more details).
    3. Position id ndarray: shape [batch, seqLen] with positions in the sentence.
    4. Attention_mask ndarray: shape [batch, seqLen] with indices in [0, 1].
       It's a mask to be used if the input sequence length is smaller than seqLen in
       the current batch.
    Output is a list which contains:
    1. The states of BERT layer.
    2. The pooled output which processes the hidden state of the last layer with regard to the first
    token of the sequence. This would be useful for segment-level tasks.

    # Arguments
    n_block: block number
    n_head: head number
    intermediate_size: The size of the "intermediate" (i.e., feed-forward)
    hidden_drop: The dropout probability for all fully connected layers
    attn_drop: drop probability of attention
    initializer_ranger: weight initialization range
    output_all_block: whether output all blocks' output
    embedding_layer: embedding layer
    input_shape: input shape
    """

    def __init__(self, n_block, n_head, intermediate_size, hidden_drop, attn_drop,
                 initializer_range, output_all_block, embedding_layer,
                 input_shape, bigdl_type="float"):
        self.hidden_drop = hidden_drop
        self.attn_drop = attn_drop
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.output_all_block = output_all_block
        self.bigdl_type = bigdl_type
        self.seq_len = input_shape[0][0]
        self.initializer_range = initializer_range
        self.bidirectional = True
        self.n_block = n_block

        word_input = Input(shape=input_shape[0])
        token_type_input = Input(shape=input_shape[1])
        position_input = Input(shape=input_shape[2])
        attention_mask = Input(shape=input_shape[3])

        e = embedding_layer([word_input, token_type_input, position_input])
        self.hidden_size = e.get_output_shape()[-1]
        extended_attention_mask = (- attention_mask + 1.0) * -10000.0

        next_input = e
        model_output = [None] * n_block
        model_output[0] = self.block(next_input, self.hidden_size, extended_attention_mask)

        for _ in range(n_block - 1):
            output = self.block(model_output[_], self.hidden_size, extended_attention_mask)
            model_output[_ + 1] = output

        pooler_output = self.pooler(model_output[-1], self.hidden_size)

        if output_all_block:
            model_output.append(pooler_output)
            model = Model([word_input, token_type_input, position_input, attention_mask],
                          model_output)
        else:
            model = Model([word_input, token_type_input, position_input, attention_mask],
                          [model_output[-1], pooler_output])
        self.value = model.value

    def projection_layer(self, output_size):
        return Dense(output_size, "normal", (0.0, self.initializer_range))

    def build_input(self, input_shape):
        if any(not isinstance(i, list) and not isinstance(i, tuple) for i in input_shape) \
                and len(input_shape) != 4:
            raise TypeError('BERT input must be a list of 4 ndarray (consisting of input'
                            ' sequence, sequence positions, segment id, attention mask)')
        inputs = [Input(list(shape)) for shape in input_shape]
        return (- inputs[-1] + 1.0) * -10000.0, inputs[:-1], inputs

    def gelu(self, x):
        y = x / math.sqrt(2.0)
        e = auto.erf(y)
        y = x * 0.5 * (e + 1.0)
        return y

    @classmethod
    def init(cls, vocab=40990, hidden_size=768, n_block=12, n_head=12,
             seq_len=512, intermediate_size=3072, hidden_drop=0.1,
             attn_drop=0.1, initializer_range=0.02, output_all_block=True,
             bigdl_type="float"):
        """
        vocab: vocabulary size of training data, default is 40990
        hidden_size: size of the encoder layers, default is 768
        n_block: block number, default is 12
        n_head: head number, default is 12
        seq_len: max sequence length of training data, default is 77
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        hidden_drop: drop probability of full connected layers, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        initializer_ranger: weight initialization range, default is 0.02
        output_all_block: whether output all blocks' output, default is True
        """
        word_input = Input(shape=(seq_len,))
        token_type_input = Input(shape=(seq_len,))
        position_input = Input(shape=(seq_len,))
        word_embedding = Embedding(vocab, hidden_size, input_length=seq_len,
                                   weights=np.random.normal(0.0, initializer_range,
                                                            (vocab, hidden_size)))(word_input)
        position_embedding = Embedding(seq_len, hidden_size, input_length=seq_len,
                                       weights=np.random.normal(0.0, initializer_range,
                                                                (seq_len, hidden_size)))(
            position_input)
        token_type_embedding = Embedding(2, hidden_size, input_length=seq_len,
                                         weights=np.random.normal(0.0, initializer_range,
                                                                  (2, hidden_size)))(
            token_type_input)
        embedding = word_embedding + position_embedding + token_type_embedding

        w = auto.Parameter(shape=(1, hidden_size),
                           init_weight=np.ones((1, hidden_size), dtype=bigdl_type))
        b = auto.Parameter(shape=(1, hidden_size),
                           init_weight=np.zeros((1, hidden_size), dtype=bigdl_type))
        after_norm = layer_norm(embedding, w, b, 1e-12)
        h = Dropout(hidden_drop)(after_norm)

        embedding_layer = Model([word_input, token_type_input, position_input], h)
        shape = ((seq_len,), (seq_len,), (seq_len,), (1, 1, seq_len))

        return BERT(n_block, n_head, intermediate_size, hidden_drop, attn_drop, initializer_range,
                    output_all_block, embedding_layer, input_shape=shape)

    @staticmethod
    def init_from_existing_model(path, weight_path=None, input_seq_len=-1.0, hidden_drop=-1.0,
                                 attn_drop=-1.0, output_all_block=True, bigdl_type="float"):
        """
        Load an existing BERT model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jlayer = callZooFunc(bigdl_type, "loadBERT", path, weight_path, input_seq_len,
                             hidden_drop, attn_drop, output_all_block)

        model = Layer(jvalue=jlayer, bigdl_type=bigdl_type)
        model.__class__ = BERT
        return model
