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


def tfrs_model(model):
    # TODO: to friesian estimator
    from tensorflow_recommenders.tasks import base
    import tensorflow as tf
    import warnings
    import tensorflow_recommenders as tfrs

    if not isinstance(model, tfrs.Model):
        return model
    attr = model.__dict__
    task_dict = dict()
    for k, v in attr.items():
        if isinstance(v, base.Task):
            task_dict[k] = v

    for k, v in task_dict.items():
        try:
            v._loss.reduction = tf.keras.losses.Reduction.SUM
        except:
            warnings.warn("Model task " + k + " has no attribute _loss, please use "
                                              "`tf.keras.losses.Reduction.SUM` or "
                                              "`tf.keras.losses.Reduction.NONE` for "
                                              "loss reduction in this task if the "
                                              "Estimator raise an error.")

    return model
