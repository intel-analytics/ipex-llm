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

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import config
from pathlib import Path
import fsspec
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

@staticmethod
def test():
    print("test")

Dataset.load_from_disk = test

# TODO: add fs argument if needed
def load_from_disk(dataset_path: str, fs: None, keep_in_memory: Optional[bool] = None) -> Union[Dataset, DatasetDict]:
    fs = fsspec.filesystem("file")
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix()):
        return Dataset.load_from_disk(dataset_path, fs, keep_in_memory=keep_in_memory)
    elif fs.isfile(Path(dataset_path, config.DATASETDICT_JSON_FILENAME).as_posix()):
        return DatasetDict.load_from_disk(dataset_path, fs, keep_in_memory=keep_in_memory)
    else:
        raise FileNotFoundError(
            f"Directory {dataset_path} is neither a dataset directory nor a dataset dict directory."
        )
