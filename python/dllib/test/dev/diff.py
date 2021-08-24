#!/usr/bin/env python


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

import os
import re
from os.path import isfile, join

scala_nn_root = "./spark/dl/src/main/scala/com/intel/analytics/bigdl/nn/"
python_nn_root = "./pyspark/bigdl/nn/"

scala_to_python = {"Graph": "Model"}


def extract_scala_class(class_path):
    exclude_key_words = set(["*", "abstract", "Const", "Fill", "Shape",
                             "SplitAndSelect", "StrideSlice", "Scheduler",
                             "StaticGraph", "DynamicGraph", "DynamicContainer",
                             "SplitHeads", "CombineHeads", "VectorProduct",
                             "MaskHead", "MaskPostProcessor", "BoxHead", "BoxPostProcessor",
                             "RegionProposal", "ProposalPostProcessor"])   # noqa
    include_key_words = set(["Module", "Criterion", "Container", "Cell", "TensorNumeric"])  # noqa
    content = "\n".join([line for line in open(class_path).readlines() if all([key not in line for key in exclude_key_words])])  # noqa
    match = re.search(r"class ([\w]+)[^{]+", content)
    if match and any([key in match.group() for key in include_key_words]):
        return match.group(1)
    else:
        return None


def get_scala_classes(scala_nn_root):
    raw_result = [extract_scala_class(join(scala_nn_root, name)) for name in os.listdir(scala_nn_root) if isfile(join(scala_nn_root, name))]  # noqa
    return set(
        class_name for class_name in raw_result if class_name is not None)


def get_python_classes(nn_root):
    exclude_classes = set(["Criterion"])
    raw_classes = []
    python_nn_layers = [join(nn_root, name) for name in os.listdir(nn_root) if isfile(join(nn_root, name)) and name.endswith('py') and "__" not in name]  # noqa
    for p in python_nn_layers:
        with open(p) as f:
            raw_classes.extend([line for line in f.readlines() if line.startswith("class")])  # noqa
    classes = [name.split()[1].split("(")[0]for name in raw_classes]
    return set([name for name in classes if name not in exclude_classes])


scala_layers = get_scala_classes(scala_nn_root)
python_layers = get_python_classes(python_nn_root)

print("scala_layers: {0}, {1}".format(len(scala_layers), scala_layers))
print("\n")
print("python_layers: {0}, {1}".format(len(python_layers), python_layers))

print("In scala, not in python: "),
diff_count = 0
for name in scala_layers:
    if name not in python_layers and scala_to_python[name] not in python_layers:
        print(name),
        diff_count += 1

if diff_count > 0:
    raise Exception("There are difference between python and scala")
