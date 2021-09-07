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

from zoo.tfpark.text.estimator import *


def make_bert_squad_model_fn(optimizer):
    def _bert_squad_model_fn(features, labels, mode, params):
        import tensorflow as tf
        from zoo.tfpark import ZooOptimizer
        final_hidden = bert_model(features, labels, mode, params).get_sequence_output()
        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        output_weights = tf.get_variable(
            "cls/squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])
        unstacked_logits = tf.unstack(logits, axis=0)
        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        if mode == tf.estimator.ModeKeys.TRAIN:
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = labels["start_positions"]
            end_positions = labels["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0
            train_op = ZooOptimizer(optimizer).minimize(total_loss)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              train_op=train_op, loss=total_loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": features["unique_ids"],
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError("Currently only TRAIN and PREDICT modes are supported. "
                             "SQuAD uses a separate script for EVAL")

    return _bert_squad_model_fn


class BERTSQuAD(BERTBaseEstimator):
    """
    A pre-built TFEstimator that that takes the hidden state of the final encoder layer of BERT
    to perform training and prediction on SQuAD dataset.
    The Stanford Question Answering Dataset (SQuAD) is a popular question answering
    benchmark dataset.

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
    def __init__(self, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None):
        super(BERTSQuAD, self).__init__(
            model_fn=make_bert_squad_model_fn(optimizer),
            bert_config_file=bert_config_file,
            init_checkpoint=init_checkpoint,
            use_one_hot_embeddings=use_one_hot_embeddings,
            model_dir=model_dir)
