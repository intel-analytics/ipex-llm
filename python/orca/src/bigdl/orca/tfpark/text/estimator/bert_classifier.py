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


def make_bert_classifier_model_fn(optimizer):
    def _bert_classifier_model_fn(features, labels, mode, params):
        """
        Model function for BERTClassifier.

        :param features: Dict of feature tensors. Must include the key "input_ids".
        :param labels: Label tensor for training.
        :param mode: 'train', 'eval' or 'infer'.
        :param params: Must include the key "num_classes".
        :return: tf.estimator.EstimatorSpec.
        """
        import tensorflow as tf
        from bigdl.orca.tfpark import ZooOptimizer
        output_layer = bert_model(features, labels, mode, params).get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [params["num_classes"], hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [params["num_classes"]], initializer=tf.zeros_initializer())
        with tf.variable_scope("loss"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=probabilities)
            else:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                one_hot_labels = tf.one_hot(labels, depth=params["num_classes"], dtype=tf.float32)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                loss = tf.reduce_mean(per_example_loss)
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=probabilities,
                                                      loss=loss)
                else:
                    train_op = ZooOptimizer(optimizer).minimize(loss)
                    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)
    return _bert_classifier_model_fn


class BERTClassifier(BERTBaseEstimator):
    """
    A pre-built TFEstimator that takes the hidden state of the first token of BERT
    to do classification.

    :param num_classes: Positive int. The number of classes to be classified.
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
    def __init__(self, num_classes, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None):
        super(BERTClassifier, self).__init__(
            model_fn=make_bert_classifier_model_fn(optimizer),
            bert_config_file=bert_config_file,
            init_checkpoint=init_checkpoint,
            use_one_hot_embeddings=use_one_hot_embeddings,
            model_dir=model_dir,
            num_classes=num_classes)
