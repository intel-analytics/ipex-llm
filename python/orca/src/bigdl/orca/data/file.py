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

from bigdl.dllib.utils.file_utils import callZooFunc


def open_text(path):
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
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        lines = data["Body"].read().decode("utf-8").strip().split("\n")
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        lines = []
        for line in open(path):
            lines.append(line)
    return [line.strip() for line in lines]


def open_image(path):
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
            return Image.open(f)
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(data["Body"].read()))
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        return Image.open(path)


def load_numpy(path):
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
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return np.load(BytesIO(data["Body"].read()))
    else:  # Local path
        if path.startswith("file://"):
            path = path[len("file://"):]
        return np.load(path)


def exists(path):
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
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        try:
            s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise ex
        return True
    elif path.startswith("hdfs://"):
        return callZooFunc("float", "exists", path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        return os.path.exists(path)


def makedirs(path):
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
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        return s3_client.put_object(Bucket=bucket, Key=key, Body='')
    elif path.startswith("hdfs://"):
        callZooFunc("float", "mkdirs", path)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        return os.makedirs(path)


def write_text(path, text):
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
            f.close()
            return result
    elif path.startswith("s3"):   # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        return s3_client.put_object(Bucket=bucket, Key=key, Body=text)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        with open(path, 'w') as f:
            result = f.write(text)
            f.close()
            return result


def put_local_dir_to_remote(local_dir, remote_dir):
    if remote_dir.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        host_port = remote_dir.split("://")[1].split("/")[0].split(":")
        classpath = subprocess.Popen(["hadoop", "classpath", "--glob"],
                                     stdout=subprocess.PIPE).communicate()[0]
        os.environ["CLASSPATH"] = classpath.decode("utf-8")
        fs = pa.hdfs.connect(host=host_port[0], port=int(host_port[1]))
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
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = remote_dir.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        for file in os.listdir(local_dir):
            with open(os.path.join(local_dir, file), "rb") as f:
                s3_client.upload_fileobj(f, Bucket=bucket, Key=prefix+'/'+file)
    else:
        if remote_dir.startswith("file://"):
            remote_dir = remote_dir[len("file://"):]
        from distutils.dir_util import copy_tree
        copy_tree(local_dir, remote_dir)
