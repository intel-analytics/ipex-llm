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

from bigdl.util.common import get_node_and_core_number
from zoo.tfpark.estimator import *
from bert import modeling


def bert_model(features, labels, mode, params):
    """
    Return an instance of BertModel and one can take its different outputs to
    perform specific tasks.
    """
    input_ids = features["input_ids"]
    if "input_mask" in features:
        input_mask = features["input_mask"]
    else:
        input_mask = None
    if "token_type_ids" in features:
        token_type_ids = features["token_type_ids"]
    else:
        token_type_ids = None
    bert_config = modeling.BertConfig.from_json_file(params["bert_config_file"])
    model = modeling.BertModel(
        config=bert_config,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        use_one_hot_embeddings=params["use_one_hot_embeddings"])
    tvars = tf.trainable_variables()
    if params["init_checkpoint"]:
        assignment_map, initialized_variable_names = \
            modeling.get_assignment_map_from_checkpoint(tvars, params["init_checkpoint"])
        tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)
    return model


def bert_input_fn(rdd, max_seq_length, batch_size, labels=None,
                  features={"input_ids", "input_mask", "token_type_ids"}):
    """
    Takes an RDD to create the input function for BERT related TFEstimators.
    For training and evaluation, each element in rdd should be a tuple: (dict of features, label).
    For prediction, each element in rdd should be a dict of features.
    """
    assert features.issubset({"input_ids", "input_mask", "token_type_ids"})
    features_dict = {}
    for feature in features:
        features_dict[feature] = (tf.int32, [max_seq_length])
    if labels is None:
        labels = (tf.int32, [])

    def input_fn(mode):
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
            return TFDataset.from_rdd(rdd,
                                      features=features_dict,
                                      labels=labels,
                                      batch_size=batch_size)
        else:
            node_num, core_num = get_node_and_core_number()
            return TFDataset.from_rdd(rdd,
                                      features=features_dict,
                                      batch_per_thread=batch_size // (node_num * core_num))
    return input_fn


class BERTBaseEstimator(TFEstimator):
    """
    The base class for BERT related TFEstimators.
    Common arguments:
    bert_config_file, init_checkpoint, use_one_hot_embeddings, optimizer, model_dir.

    For its subclass:
    - One can add additional arguments and access them via `params`.
    - One can utilize `_bert_model` to create model_fn and `bert_input_fn` to create input_fn.
    """
    def __init__(self, model_fn, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None, **kwargs):
        params = {"bert_config_file": bert_config_file,
                  "init_checkpoint": init_checkpoint,
                  "use_one_hot_embeddings": use_one_hot_embeddings}
        for k, v in kwargs.items():
            params[k] = v
        super(BERTBaseEstimator, self).__init__(
            model_fn=model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
            params=params)
