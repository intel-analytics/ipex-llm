#
# Copyright 2016 The BigDL Authors.
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

import tensorflow as tf
from tensorflow.keras.layers import Embedding as TFEmbedding


class Embedding(TFEmbedding):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):

        '''
        This is a slightly modified version of tf.keras.Embedding,
        which only apply regularizer to the output of the embedding
        layers, such that the gradient to embeddings is sparse.

        :param input_sample: torch.Tensor or a list for the model tracing.
        :param file_path: The path to save onnx model file.
        :param sess_options: ortsess options in ort.SessionOptions type
        :param **kwargs: will be passed to torch.onnx.export function.

        :param input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        :param output_dim: Integer. Dimension of the dense embedding.
        :param embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        :param regularizer: Regularizer function applied to
            the output tensor after looking up the `embeddings` matrix.
        :param embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        :param mask_zero: Boolean, whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using recurrent layers
            which may take variable length input.
            If this is `True`, then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).
        param: input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
        '''

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         embeddings_initializer=embeddings_initializer,
                         embeddings_regularizer=None,
                         embeddings_constraint=embeddings_constraint,
                         activity_regularizer=regularizer,
                         mask_zero=mask_zero,
                         input_length=input_length,
                         **kwargs)
