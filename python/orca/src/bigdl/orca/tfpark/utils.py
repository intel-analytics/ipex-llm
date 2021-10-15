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
from bigdl.orca.tfpark.tfnet import TFNet
from bigdl.orca.tfpark.tf_optimizer import BigDLMetric, TFModel
from bigdl.dllib.keras import metrics as zmetrics


def to_bigdl_metric(metric):
    metric = metric.lower()
    if metric == "accuracy" or metric == "acc":
        return zmetrics.Accuracy()
    elif metric == "top5accuracy" or metric == "top5acc":
        return zmetrics.Top5Accuracy()
    elif metric == "mae":
        from bigdl.dllib.optim.optimizer import MAE
        return MAE()
    elif metric == "auc":
        return zmetrics.AUC()
    elif metric == "treennaccuracy":
        from bigdl.dllib.optim.optimizer import TreeNNAccuracy
        return TreeNNAccuracy()
    else:
        raise TypeError("Unsupported metric: %s" % metric)


def evaluate_string_metrics(*,
                            sess,
                            string_metrics,
                            dataset,
                            inputs,
                            targets=None,
                            outputs=None,
                            loss=None,
                            ):

    metrics = {}
    for i, metric in enumerate(string_metrics):
        if metric == "loss":
            assert loss is not None, "loss tensor should not be None if one of the metrics is loss"
            metrics["loss"] = loss
        else:
            assert outputs is not None, "outputs should not be None if non loss metrics exists"
            assert targets is not None, "targets should not be None if non loss metrics exists"

            method = to_bigdl_metric(metric)
            metrics[metric] = BigDLMetric(method,
                                          outputs,
                                          targets)
    result = evaluate_metrics(inputs, sess, dataset, metrics)
    return result


def evaluate_metrics(inputs, sess, dataset, metrics):
    import tensorflow as tf
    if dataset.batch_per_thread > 0:
        batch_size = dataset.batch_per_thread * dataset.get_num_partitions()
    else:
        batch_size = dataset.batch_size

    real_batch_size = tf.shape(inputs[0])[0]

    outputs, eval_methods = TFModel._process_metrics(inputs[0].graph,
                                                     metrics=metrics,
                                                     real_batch_size=real_batch_size)

    tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

    results = tfnet.evaluate(dataset, batch_size, eval_methods)
    final_result = dict([(r.method, r.result) for r in results])
    return final_result
