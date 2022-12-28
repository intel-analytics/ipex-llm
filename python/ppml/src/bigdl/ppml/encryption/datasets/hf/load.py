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
from bigdl.ppml.encryption.torch.models import opener
from datasets import config
from cryptography.fernet import Fernet
from pathlib import Path
import fsspec
import json
from io import BytesIO
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.utils.info_utils import is_small_dataset
from datasets.splits import Split
# TODO: This may incur error, the encrypted filesize is larger
from datasets.utils.file_utils import estimate_dataset_size
# TODO: clean
from datasets.table import (
    InMemoryTable,
    MemoryMappedTable,
    Table,
    concat_tables,
    embed_table_storage,
    list_table_cache_files,
    table_cast,
    table_iter,
    table_visitor,
)
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

class encrypt_reader_opener(opener):
    def __init__(self, name, mode, key):
        self.key = Fernet(key)
        opened_file = open(name, mode)
        decrypted_content = self.key.decrypt(opened_file.read())
        opened_file.close()
        buf = BytesIO()
        buf.write(decrypted_content)
        buf.seek(0)
        super(encrypt_reader_opener, self).__init__(buf)

# TODO: delete comment
# We knew that file is a path
# Review later
def _open_encrypt_file_with_key(file, mode, key):
    return encrypt_reader_opener(file, mode, key)


def decrypt_file_to_pa_buffer(filename, key):
    decryptor = Fernet(key)
    opened_file = open(filename, 'rb')
    decrypted_content = decryptor.decrypt(opened_file.read())
    opened_file.close()
    buf = pa.py_buffer(decrypted_content)
    return buf


@staticmethod
def customized_load(dataset_path: str, key:str, fs=None, keep_in_memory: Optional[bool] = None) -> "Dataset":
    print("customized load") # TODO: delete later
    # TODO: delete fs later
    fs = fsspec.filesystem("file") if fs is None else fs
    dataset_dict_json_path = Path(dataset_path, config.DATASETDICT_JSON_FILENAME).as_posix()
    dataset_info_path = Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix()
    if not fs.isfile(dataset_info_path) and fs.isfile(dataset_dict_json_path):
        raise FileNotFoundError(
           f"No such file or directory: '{dataset_info_path}'. Expected to load a Dataset object, but got a DatasetDict. Please use datasets.load_from_disk instead."
        )

    # TODO: use own own opener later here
    # with open(Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), encoding="utf-8") as state_file:
    #     state = json.load(state_file)
    with _open_encrypt_file_with_key(Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), 'rb', key) as state_file:
        state = json.load(state_file)
    print(state, flush=True)
    # with open(Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix(), encoding="utf-8") as dataset_info_file:
    #     dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))
    with _open_encrypt_file_with_key(Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix(), 'rb', key) as dataset_info_file:
        dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_size = estimate_dataset_size(
        Path(dataset_path, data_file["filename"]) for data_file in state["_data_files"]
    )
    keep_in_memory = keep_in_memory if keep_in_memory is not None else is_small_dataset(dataset_size)
    # TODO: change others later
    table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable
    # arrow_table = concat_tables(
    #     table_cls.from_file(Path(dataset_path, data_file["filename"]).as_posix())
    #     for data_file in state["_data_files"]
    # )
    # TODO: change later
    table_cls = InMemoryTable
    arrow_table = concat_tables(
        table_cls.from_buffer(decrypt_file_to_pa_buffer(
            Path(dataset_path, data_file["filename"]).as_posix(), key))
        for data_file in state["_data_files"]
    )

    # TODO: close the pa.buffer opened above
    # TODO: check MemoryMappedTable
    split = state["_split"]
    split = Split(split) if split is not None else split

    dataset = Dataset(
        arrow_table=arrow_table,
        info=dataset_info,
        split=split,
        fingerprint=state["_fingerprint"],
    )

    format = {
        "type": state["_format_type"],
        "format_kwargs": state["_format_kwargs"],
        "columns": state["_format_columns"],
        "output_all_columns": state["_output_all_columns"],
    }
    dataset = dataset.with_format(**format)

    return dataset


Dataset.load_from_disk = customized_load

# TODO: add fs argument if needed
def load_from_disk(dataset_path: str, key: str, fs=None, keep_in_memory: Optional[bool] = None) -> Union[Dataset, DatasetDict]:
    fs = fsspec.filesystem("file")
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix()):
        return Dataset.load_from_disk(dataset_path, key, fs, keep_in_memory=keep_in_memory)
    elif fs.isfile(Path(dataset_path, config.DATASETDICT_JSON_FILENAME).as_posix()):
        return DatasetDict.load_from_disk(dataset_path, fs, keep_in_memory=keep_in_memory)
    else:
        raise FileNotFoundError(
            f"Directory {dataset_path} is neither a dataset directory nor a dataset dict directory."
        )
