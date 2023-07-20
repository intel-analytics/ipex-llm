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
import subprocess
import logging
import shutil
import functools
import glob
from distutils.dir_util import copy_tree
from bigdl.dllib.utils.log4Error import invalidOperationError
from bigdl.dllib.utils import file_utils

from typing import TYPE_CHECKING, List, Callable, Union, Optional
if TYPE_CHECKING:
    from PIL.JpegImagePlugin import JpegImageFile
    from numpy import ndarray
    from typing import List

logger = logging.getLogger(__name__)


def open_text(path: str) -> List[str]:
    """

    Read a text file to list of lines. It supports local, hdfs, s3 file systems.

    :param path: text file path
    :return: list of lines
    """
    # Return a list of lines
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            lines = f.read().decode("utf-8").strip().split("\n")
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        lines = data["Body"].read().decode("utf-8").strip().split("\n")
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        lines = []
        with open(path) as f:
            for line in f:
                lines.append(line)
    return [line.strip() for line in lines]


def open_image(path: str) -> "JpegImageFile":
    """

    Open a image file. It supports local, hdfs, s3 file systems.

    :param path: an image file path
    :return: An :py:class:`~PIL.Image.Image` object.
    """
    from PIL import Image
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            image = Image.open(f)
            image.load()
        return image
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        with BytesIO(data["Body"].read()) as f:
            return Image.open(f)
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        with open(path, "rb") as f:
            image = Image.open(f)
            image.load()
        return image


def load_numpy(path: str) -> "ndarray":
    """

    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
    It supports local, hdfs, s3 file systems.

    :param path: file path
    :return: array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.
    """
    import numpy as np
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            return np.load(f)
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return np.load(BytesIO(data["Body"].read()))
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        return np.load(path)


def exists(path: str) -> bool:
    """

    Check if a path exists or not. It supports local, hdfs, s3 file systems.

    :param path: file or directory path string.
    :return: if path exists or not.
    """
    if path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        try:
            s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':  # type:ignore
                return False
            invalidOperationError(False, str(ex), cause=ex)
        return True
    elif path.startswith("hdfs://"):
        return file_utils.exists(path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        return os.path.exists(path)


def listdir(path: str) -> List[str]:
    """

    List file and directory names in a directory.
    It supports local, hdfs. S3 file systems is not implemented.

    :param path: directory path string.
    """
    if path.startswith("s3"):  # s3://bucket/file_path
        invalidOperationError(False, "not implement")
        return []
    elif path.startswith("hdfs://"):
        return file_utils.get_file_list(path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        return os.listdir(path)


def makedirs(path: str) -> None:
    """

    Make a directory with creating intermediate directories.
    It supports local, hdfs, s3 file systems.

    :param path: directory path string to be created.
    """
    if path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        return s3_client.put_object(Bucket=bucket, Key=key, Body='')
    elif path.startswith("hdfs://"):
        return file_utils.mkdirs(path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        os.makedirs(path)


def rmdir(path: str) -> None:
    """

    Remove a directory.
    It supports local, hdfs. S3 file systems is not implemented.

    :param path: directory path string to be removed.
    """
    if path.startswith("s3"):  # s3://bucket/file_path
        invalidOperationError(False, "not implement")
    elif path.startswith("hdfs://"):
        file_utils.rmdir(path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        os.rmdir(path)


def write_text(path: str, text: str) -> int:
    """

    Write text to a file. It supports local, hdfs, s3 file systems.

    :param path: file path
    :param text: text string
    :return: number of bytes written or AWS response(s3 file systems)
    """
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'wb') as f:
            result = f.write(text.encode('utf-8'))
            return result
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        return s3_client.put_object(Bucket=bucket, Key=key, Body=text)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        with open(path, 'w') as f:
            result = f.write(text)
            return result


def is_file(path: str) -> bool:
    """

    Check if a path is file or not. It supports local, hdfs, s3 file systems.

    :param path: path string.
    :return: if path is a file.
    """
    if path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        try:
            dir_key = key + '/'
            resp1 = s3_client.list_objects(Bucket=bucket, Prefix=key, Delimiter='/', MaxKeys=1)
            if 'Contents' in resp1:
                resp2 = s3_client.list_objects(Bucket=bucket, Prefix=dir_key,
                                               Delimiter='/', MaxKeys=1)
                return not ('Contents' in resp2)
            else:
                return False
        except Exception as ex:
            invalidOperationError(False, str(ex), cause=ex)
            return False
    elif path.startswith("hdfs://"):
        return file_utils.is_file(path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        from pathlib import Path
        return Path(path).is_file()


def put_local_dir_to_remote(local_dir: str, remote_dir: str):
    if remote_dir.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        host_port = remote_dir.split("://")[1].split("/")[0].split(":")
        with subprocess.Popen(["hadoop", "classpath", "--glob"], stdout=subprocess.PIPE) as p:
            classpath = p.communicate()[0]
        os.environ["CLASSPATH"] = classpath.decode("utf-8")
        if len(host_port) > 1:
            fs = pa.hdfs.connect(host=host_port[0], port=int(host_port[1]))
        else:
            fs = pa.hdfs.connect(host=host_port[0])
        if not fs.exists(remote_dir):
            fs.mkdir(remote_dir)
        for file in os.listdir(local_dir):
            with open(os.path.join(local_dir, file), "rb") as f:
                fs.upload(os.path.join(remote_dir, file), f)
    elif remote_dir.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_dir.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        for file in os.listdir(local_dir):
            with open(os.path.join(local_dir, file), "rb") as f:
                s3_client.upload_fileobj(f, Bucket=bucket, Key=prefix + '/' + file)
    else:
        if remote_dir.startswith("file://"):
            remote_dir = remote_dir[len("file://"):]
        copy_tree(local_dir, remote_dir)


def put_local_dir_tree_to_remote(local_dir: str, remote_dir: str,
                                 over_write: Optional[bool] = False):
    if remote_dir.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.put_local_dir_tree_to_remote(local_dir=local_dir,
                                                       remote_dir=remote_dir,
                                                       over_write=over_write)
    elif remote_dir.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_dir.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        local_files = [os.path.join(dirpath, f)
                       for (dirpath, dirnames, filenames) in os.walk(local_dir)
                       for f in filenames]
        for file in local_files:
            try:
                with open(file, "rb") as f:
                    s3_client.upload_fileobj(f, Bucket=bucket,
                                             Key=prefix + '/' + file[len(local_dir) + 1:])
            except Exception as e:
                logger.error('cannot upload file to s3: {}'.format(str(e)))
                return -1
        return 0
    else:
        if remote_dir.startswith("file://"):
            remote_dir = remote_dir[len("file://"):]
        try:
            copy_tree(local_dir, remote_dir)
        except Exception as e:
            logger.warning(str(e))
            return -1
        return 0


def put_local_file_to_remote(local_path: str, remote_path: str,
                             filemode: Optional[int] = None,
                             over_write: Optional[bool] = False) -> int:
    if remote_path.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.put_local_file_to_remote(local_path=local_path,
                                                   remote_path=remote_path,
                                                   over_write=over_write)
    elif remote_path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        try:
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key).client('s3')
            path_parts = remote_path.split("://")[1].split('/')
            bucket = path_parts.pop(0)
            prefix = "/".join(path_parts)
            with open(local_path, "rb") as f:
                s3_client.upload_fileobj(f, Bucket=bucket, Key=prefix)
        except Exception as e:
            logger.error("Cannot upload file {} to {}: error: {}"
                         .format(local_path, remote_path, str(e)))
            return -1
        return 0
    else:
        if remote_path.startswith("file://"):
            remote_path = remote_path[len("file://"):]
        try:
            shutil.copy(local_path, remote_path)
            if filemode:
                os.chmod(remote_path, filemode)
        except Exception as e:
            logger.error("Cannot upload file {} to {}: error: {}"
                         .format(local_path, remote_path, str(e)))
            return -1
        return 0


def put_local_files_with_prefix_to_remote(local_path_prefix: str, remote_dir: str,
                                          over_write: Optional[bool] = False) -> int:
    file_list = glob.glob(local_path_prefix + "*")
    if remote_dir.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.put_local_files_with_prefix_to_remote(local_path_prefix=local_path_prefix,
                                                                remote_dir=remote_dir,
                                                                over_write=over_write)
    elif remote_dir.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_dir.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        local_dir = os.path.dirname(local_path_prefix)
        try:
            [s3_client.upload_file(os.path.join(local_dir, file), bucket,
                                   os.path.join(prefix, file)) for file in file_list]
        except Exception as e:
            logger.error(str(e))
            return -1
        return 0
    else:
        if remote_dir.startswith("file://"):
            remote_dir = remote_dir[len("file://"):]
        try:
            [shutil.copy(local_file, remote_dir) for local_file in file_list]
        except Exception as e:
            logger.error(str(e))
            return -1
        return 0


def get_remote_file_to_local(remote_path: str, local_path: str,
                             over_write: Optional[bool] = False) -> int:
    if remote_path.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.get_remote_file_to_local(remote_path=remote_path,
                                                   local_path=local_path,
                                                   over_write=over_write)
    elif remote_path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        try:
            s3_client.download_file(bucket, key, local_path)
            return 0
        except Exception as e:
            print(str(e))
            return -1
    else:
        if remote_path.startswith("file://"):
            remote_path = remote_path[len("file://"):]
        shutil.copy(remote_path, local_path)
        return 0


def get_remote_dir_to_local(remote_dir: str, local_dir: str,
                            over_write: Optional[bool] = False) -> int:
    if remote_dir.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.get_remote_dir_tree_to_local(remote_dir=remote_dir,
                                                       local_dir=local_dir,
                                                       over_write=over_write)
    elif remote_dir.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_dir.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
            keys = [item['Key'] for item in response['Contents']]
            [s3_client.download_file(bucket, key,
                                     os.path.join(local_dir,
                                                  os.path.basename(keys)))  # type: ignore
             for key in keys]
        except Exception as e:
            invalidOperationError(False, str(e), cause=e)
        return 0
    else:
        if remote_dir.startswith("file://"):
            remote_dir = remote_dir[len("file://"):]
        copy_tree(remote_dir, local_dir)
        return 0


def get_remote_files_with_prefix_to_local(remote_path_prefix: str,
                                          local_dir: str,
                                          over_write: Optional[bool] = False) -> Union[str, int]:
    prefix = os.path.basename(remote_path_prefix)
    if remote_path_prefix.startswith("hdfs"):  # hdfs://url:port/file_path
        return file_utils.get_remote_files_with_prefix_to_local(
            remote_path_prefix=remote_path_prefix,
            local_dir=local_dir,
            over_write=over_write)
    elif remote_path_prefix.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = remote_path_prefix.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            keys = [item['Key'] for item in response['Contents']]
            [s3_client.download_file(bucket, key,
                                     os.path.join(local_dir,
                                                  os.path.basename(keys)))  # type: ignore
             for key in keys]
        except Exception as e:
            invalidOperationError(False, str(e), cause=e)
    return os.path.join(local_dir, prefix)


def enable_multi_fs_load_static(load_func: Callable) -> Callable:
    """
    Enable loading file or directory in multiple file systems.
    It supports local, hdfs, s3 file systems.
    Note: this decorator is different from dllib decorator @enable_multi_fs_load_static.
    This decorator can load on each worker while @enable_multi_fs_load can only load on driver.

    :param load_func: load file or directory function
    :return: load file or directory function for the specific file system
    """
    @functools.wraps(load_func)
    def fs_load(path, *args, **kwargs):
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(path):
            return load_func(path, *args, **kwargs)
        else:
            import uuid
            import tempfile
            from bigdl.dllib.utils.file_utils import append_suffix
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path.strip("/").split("/")[-1])
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = os.path.join(tmpdir, file_name)
                if is_file(path):
                    get_remote_file_to_local(path, temp_path)
                else:
                    os.makedirs(temp_path)
                    get_remote_dir_to_local(path, temp_path)
                return load_func(temp_path, *args, **kwargs)
    return fs_load


def enable_multi_fs_save(save_func: Callable) -> Callable:

    @functools.wraps(save_func)
    def fs_save(obj, path, *args, **kwargs):
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(path):
            return save_func(obj, path, *args, **kwargs)
        else:
            import uuid
            import tempfile
            from bigdl.dllib.utils.file_utils import append_suffix
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path)
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = os.path.join(tmpdir, file_name)
                result = save_func(obj, temp_path, *args, **kwargs)
                put_local_file_to_remote(temp_path, path)
            return result

    return fs_save


def enable_multi_fs_load(load_func: Callable) -> Callable:

    @functools.wraps(load_func)
    def fs_load(obj, path, *args, **kwargs):
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(path):
            return load_func(obj, path, *args, **kwargs)
        else:
            import uuid
            import tempfile
            from bigdl.dllib.utils.file_utils import append_suffix
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path)
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = os.path.join(tmpdir, file_name)
                get_remote_file_to_local(path, temp_path)
                return load_func(obj, temp_path, *args, **kwargs)

    return fs_load
