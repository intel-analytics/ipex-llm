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
from datasets.filesystems import is_remote_filesystem, extract_path_from_uri
from bigdl.ppml.encryption.torch.models import opener
from datasets import config
from cryptography.fernet import Fernet
from pathlib import Path
import fsspec
import json
from io import BytesIO
from bigdl.nano.utils.log4Error import invalidInputError
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.utils.info_utils import is_small_dataset
from datasets.splits import Split
from datasets.utils.file_utils import estimate_dataset_size
from datasets.table import (
    InMemoryTable,
    MemoryMappedTable,
    concat_tables,
)
from typing import Optional, Union


class _opened_file(opener):
    def __init__(self, file_like):
        super(_opened_file, self).__init__(file_like)

    def __exit__(self, *args):
        # Flush is automatically done when closing the file
        self.file_like.close()


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
    
    def __exit__(self, *args):
        # Close the buffer
        self.file_like.close()

class encrypt_file_opener(opener):
    def __init__(self, file_like, key):
        self.key = Fernet(key)
        decrypted_content = self.key.decrypt(file_like.read())
        file_like.close()
        buf = BytesIO()
        buf.write(decrypted_content)
        buf.seek(0)
        super(encrypt_file_opener, self).__init__(buf)
    
    def __exit__(self, *args):
        self.file_like.close()


def _open_encrypted_file_like_with_key(path, fs, mode, key: Optional[str] = None):
    if key is not None:
        mode += 'b'
    file_like = fs.open(path, mode)
    if key is not None:
        return encrypt_file_opener(file_like, key)
    else:
        return _opened_file(file_like)


def _open_encrypt_file_with_key(file, mode, key: Optional[str] = None):
    if key is not None:
        return encrypt_reader_opener(file, mode, key)
    else:
        return open(file, encoding="utf-8")

def decrypt_file_to_pa_buffer(filename, key):
    decryptor = Fernet(key)
    opened_file = open(filename, 'rb')
    decrypted_content = decryptor.decrypt(opened_file.read())
    opened_file.close()
    buf = pa.py_buffer(decrypted_content)
    return buf


@staticmethod
def load_dict_with_decryption(dataset_dict_path: str, fs=None, key: Optional[str] = None, keep_in_memory: Optional[bool] = None) -> "DatasetDict":
    """
    Load a dataset that was previously saved using :meth:`save_to_disk` from a filesystem using either
    :class:`~filesystems.S3FileSystem` or ``fsspec.spec.AbstractFileSystem``.

    If key is not None, then dataset stored at dataset_path must be previously encrypted using the key provided
    here.  At the same time, the keep_in_memory must be set to True.

    Args:
        dataset_dict_path (:obj:`str`): Path (e.g. ``"dataset/train"``) or remote URI (e.g.
            ``"s3//my-bucket/dataset/train"``) of the dataset dict directory where the dataset dict will be loaded
            from.
        fs (:class:`~filesystems.S3FileSystem` or ``fsspec.spec.AbstractFileSystem``, optional, default ``None``):
            Instance of the remote filesystem used to download the files from.
        keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the
            dataset will not be copied in-memory unless explicitly enabled by setting
            `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
            :ref:`load_dataset_enhancing_performance` section.
        key (:obj:`str`, default ``None``): If not None, then the dataset is considered to be encrypted previously using
            this key.

    Returns:
        :class:`DatasetDict`
    """
    _check_key_compatible_with_in_mem(key, keep_in_memory)
    dataset_dict = DatasetDict()
    if is_remote_filesystem(fs):
        dest_dataset_dict_path = extract_path_from_uri(dataset_dict_path)
    else:
        fs = fsspec.filesystem("file")
        dest_dataset_dict_path = dataset_dict_path
    dataset_dict_json_path = Path(dest_dataset_dict_path, config.DATASETDICT_JSON_FILENAME).as_posix()
    dataset_info_path = Path(dest_dataset_dict_path, config.DATASET_INFO_FILENAME).as_posix()
    if fs.isfile(dataset_info_path) and not fs.isfile(dataset_dict_json_path):
        invalidInputError(False,
            f"No such file or directory: '{dataset_dict_json_path}'. Expected to load a DatasetDict object, but got a Dataset. Please use datasets.load_from_disk instead."
        )
    #with _open_encrypted_file_like_with_key(fs.open(dataset_dict_json_path, "r", key=key)) as config:
    with _open_encrypted_file_like_with_key(dataset_dict_json_path, fs, "r", key=key) as config:
        for k in json.load(config)["splits"]:
            dataset_dict_split_path = (
                dataset_dict_path.split("://")[0] + "://" + Path(dest_dataset_dict_path, k).as_posix()
                if is_remote_filesystem(fs)
                else Path(dest_dataset_dict_path, k).as_posix()
            )
            dataset_dict[k] = Dataset.load_from_disk(dataset_dict_split_path, fs, keep_in_memory=keep_in_memory)

    return dataset_dict


@staticmethod
def load_with_decryption(dataset_path: str, fs=None, keep_in_memory: Optional[bool] = None, key: Optional[str] = None) -> "Dataset":
    """
    Loads a dataset that was previously saved using :meth:`save_to_disk` from a dataset directory, or from a
    filesystem using either :class:`~filesystems.S3FileSystem` or any implementation of
    ``fsspec.spec.AbstractFileSystem``.

    If key is not None, then dataset stored at dataset_path must be previously encrypted using the key provided
    here.  At the same time, the keep_in_memory must be set to True.

    Args:
        dataset_path (:obj:`str`): Path (e.g. `"dataset/train"`) or remote URI (e.g.
            `"s3//my-bucket/dataset/train"`) of the dataset directory where the dataset will be loaded from.
        fs (:class:`~filesystems.S3FileSystem`, ``fsspec.spec.AbstractFileSystem``, optional, default ``None``):
            Instance of the remote filesystem used to download the files from.
        keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the
            dataset will not be copied in-memory unless explicitly enabled by setting
            `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
            :ref:`load_dataset_enhancing_performance` section.
        key (:obj:`str`, default ``None``): If not None, then the dataset is considered to be encrypted previously using
            this key.


    Returns:
        :class:`Dataset` or :class:`DatasetDict`:
        - If `dataset_path` is a path of a dataset directory: the dataset requested.
        - If `dataset_path` is a path of a dataset dict directory: a ``datasets.DatasetDict`` with each split.

    Example:

    ```py
    >>> ds = load_from_disk("path/to/dataset/directory", keep_in_memory=True, key=my_key)
    ```
    """
    _check_key_compatible_with_in_mem(key, keep_in_memory)
    fs = fsspec.filesystem("file") if fs is None else fs
    dataset_dict_json_path = Path(dataset_path, config.DATASETDICT_JSON_FILENAME).as_posix()
    dataset_info_path = Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix()
    if not fs.isfile(dataset_info_path) and fs.isfile(dataset_dict_json_path):
        invalidInputError(False, f"No such file or directory: '{dataset_info_path}'. Expected to load a Dataset object, but got a DatasetDict. Please use datasets.load_from_disk instead.")

    if is_remote_filesystem(fs):
        src_dataset_path = extract_path_from_uri(dataset_path)
        dataset_path = Dataset._build_local_temp_path(src_dataset_path)
        fs.download(src_dataset_path, dataset_path.as_posix(), recursive=True)

    with _open_encrypt_file_with_key(Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), 'rb', key=key) as state_file:
        state = json.load(state_file)
    with _open_encrypt_file_with_key(Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix(), 'rb', key=key) as dataset_info_file:
        dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_size = estimate_dataset_size(
        Path(dataset_path, data_file["filename"]) for data_file in state["_data_files"]
    )
 
    keep_in_memory = keep_in_memory if keep_in_memory is not None else is_small_dataset(dataset_size)
    table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable
    if key is not None:
        arrow_table = concat_tables(
            table_cls.from_buffer(decrypt_file_to_pa_buffer(
                Path(dataset_path, data_file["filename"]).as_posix(), key))
            for data_file in state["_data_files"]
        )
    else:
        arrow_table = concat_tables(
            table_cls.from_file(Path(dataset_path, data_file["filename"]).as_posix())
            for data_file in state["_data_files"]
        )



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


Dataset.load_from_disk = load_with_decryption
DatasetDict.load_from_disk = load_dict_with_decryption


def _check_key_compatible_with_in_mem(key: Union[str, None], keep_in_memory: Union[bool, None]):
    if key is not None:
        invalidInputError(keep_in_memory==True, 
                          "Currently only support InMemoryTable which \
                          requires keep_in_memory set to True")


def load_from_disk(dataset_path: str, fs=None, keep_in_memory: Optional[bool] = None, key: Optional[str] = None) -> Union[Dataset, DatasetDict]:
    """
    Loads a dataset that was previously saved using :meth:`Dataset.save_to_disk` from a dataset directory, or
    from a filesystem using either :class:`datasets.filesystems.S3FileSystem` or any implementation of
    ``fsspec.spec.AbstractFileSystem``.

    If key is not None, then dataset stored at dataset_path must be previously encrypted using the key provided
    here.  At the same time, the keep_in_memory must be set to True.

    Args:
        dataset_path (:obj:`str`): Path (e.g. `"dataset/train"`) or remote URI (e.g.
            `"s3://my-bucket/dataset/train"`) of the Dataset or DatasetDict directory where the dataset will be
            loaded from.
        fs (:class:`~filesystems.S3FileSystem` or ``fsspec.spec.AbstractFileSystem``, optional, default ``None``):
            Instance of the remote filesystem used to download the files from.
        keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the :ref:`load_dataset_enhancing_performance` section.
        key (:obj:`str`, default ``None``): If not None, then the dataset is considered to be encrypted previously using
            this key.

    Returns:
        :class:`Dataset` or :class:`DatasetDict`:
        - If `dataset_path` is a path of a dataset directory: the dataset requested.
        - If `dataset_path` is a path of a dataset dict directory: a ``datasets.DatasetDict`` with each split.
    """
    _check_key_compatible_with_in_mem(key, keep_in_memory)
    if is_remote_filesystem(fs):
        dest_dataset_path = extract_path_from_uri(dataset_path)
    else:
        fs = fsspec.filesystem("file")
        dest_dataset_path = dataset_path

    if not fs.exists(dest_dataset_path):
        invalidInputError(False, f"Directory {dataset_path} not found")
    if fs.isfile(Path(dest_dataset_path, config.DATASET_INFO_FILENAME).as_posix()):
        return Dataset.load_from_disk(dataset_path, fs, key=key, keep_in_memory=keep_in_memory)
    # TODO: test this later, check this later
    elif fs.isfile(Path(dest_dataset_path, config.DATASETDICT_JSON_FILENAME).as_posix()):
        return DatasetDict.load_from_disk(dataset_path, fs, keep_in_memory=keep_in_memory)
    else:
        invalidInputError(False,
            f"Directory {dataset_path} is neither a dataset directory nor a dataset dict directory."
        )
