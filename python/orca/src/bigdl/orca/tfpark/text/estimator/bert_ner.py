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
from bigdl.orca.tfpark.text.estimator import *


def make_bert_ner_model_fn(optimizer):
    import tensorflow as tf
    from bigdl.orca.tfpark import ZooOptimizer

    def _bert_ner_model_fn(features, labels, mode, params):
        output_layer = bert_model(features, labels, mode, params).get_sequence_output()
        if mode == tf.estimator.ModeKeys.TRAIN:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.layers.dense(output_layer, params["num_entities"])
        if mode == tf.estimator.ModeKeys.TRAIN:
            logits = tf.reshape(logits, [-1, params["num_entities"]])
            labels = tf.reshape(labels, [-1])
            mask = tf.cast(features["input_mask"], dtype=tf.float32)
            one_hot_labels = tf.one_hot(labels, depth=params["num_entities"], dtype=tf.float32)
            loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
            loss *= tf.reshape(mask, [-1])
            loss = tf.reduce_sum(loss)
            total_size = tf.reduce_sum(mask)
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            loss /= total_size
            train_op = ZooOptimizer(optimizer).minimize(loss)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              train_op=train_op, loss=loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predict)
        else:
            raise ValueError("Currently only TRAIN and PREDICT modes are supported for NER")
    return _bert_ner_model_fn


class BERTNER(BERTBaseEstimator):
    """
    A pre-built TFEstimator that takes the hidden state of the final encoder layer of BERT
    for named entity recognition based on SoftMax classification.
    Note that cased BERT models are recommended for NER.

    :param num_entities: Positive int. The number of entity labels to be classified.
    :param bert_config_file: The path to the json file for BERT configurations.
    :param init_checkpoint: The path to the initial checkpoint of the pre-trained BERT model if any.
                            Default is None.
    :param use_one_hot_embeddings: Boolean. Whether to use one-hot for word embeddings.
                                   Default is False.
    :param optimizer: The optimizer used to train the estimator. It should be an instance of
                      tf.train.Optimizer.
                      Default is None if no training is involved.
    :param model_dir: The output directory for model checkpoints to be written if any.
                      Default is None.
    """
    def __init__(self, num_entities, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None):
        super(BERTNER, self).__init__(
            model_fn=make_bert_ner_model_fn(optimizer),
            bert_config_file=bert_config_file,
            init_checkpoint=init_checkpoint,
            use_one_hot_embeddings=use_one_hot_embeddings,
            model_dir=model_dir,
            num_entities=num_entities)
