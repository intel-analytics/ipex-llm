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


def find_latest_checkpoint(model_dir, model_type="bigdl"):
    import os
    import re
    import datetime
    ckpt_path = None
    latest_version = None
    optim_prefix = None
    optim_regex = None
    if model_type == "bigdl":
        optim_regex = ".*\.[0-9]+$"
    elif model_type == "pytorch":
        optim_regex = "TorchModel[0-9a-z]*\.[0-9]+$"
    elif model_type == "tf":
        optim_regex = "TFParkTraining\.[0-9]+$"
    else:
        ValueError("Only bigdl, pytorch and tf are supported for now.")
    for (root, dirs, files) in os.walk(model_dir, topdown=True):
        temp_versions = []
        timestamps = []
        prefix = None
        for dir in dirs:
            if re.match('(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$', dir) is not None:
                try:
                    # check if dir name is date time
                    datetime.datetime.strptime(dir, '%Y-%m-%d_%H-%M-%S')
                    timestamps.append(dir)
                except:
                    continue
        if timestamps:
            start_dir = os.path.join(root, max(timestamps))
            return find_latest_checkpoint(start_dir, model_type=model_type)
        for file_name in files:
            if re.match("^optimMethod-" + optim_regex, file_name) is not None:
                file_split = file_name.split(".")
                version = int(file_split[1])
                temp_versions.append(version)
                prefix = file_split[0]
        if temp_versions:
            ckpt_path = root
            latest_version = max(temp_versions)
            optim_prefix = prefix
            break
    return ckpt_path, optim_prefix, latest_version


def convert_predict_to_xshard(prediction_rdd):
    import numpy as np
    from zoo.orca.data import SparkXShards

    def transform_predict(iter):
        predictions = list(iter)
        # list of np array
        if isinstance(predictions[0], list):
            predictions = np.array(predictions).T.tolist()
            result = [np.array(predict) for predict in predictions]
            return [{'prediction': result}]
        # np array
        else:
            return [{'prediction': np.array(predictions)}]

    return SparkXShards(prediction_rdd.mapPartitions(transform_predict))
