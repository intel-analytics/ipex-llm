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
import os

from zoo.common import get_file_list


# split list into n chunks
def chunk(lst, n):
    size = len(lst) // n
    leftovers = lst[size * n:]
    for c in range(n):
        if leftovers:
            extra = [leftovers.pop()]
        else:
            extra = []
        yield lst[c * size:(c + 1) * size] + extra


def flatten(list_input):
    if any(isinstance(i, list) for i in list_input):
        return [item for sublist in list_input for item in sublist]
    else:
        return list_input


def list_s3_file(file_path, file_type, env):
    path_parts = file_path.split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = env["AWS_ACCESS_KEY_ID"]
    secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    # file
    if os.path.splitext(file_path)[1] != '':
        return ["s3://" + file_path]
    else:
        keys = []
        resp = s3_client.list_objects_v2(Bucket=bucket,
                                         Prefix=key)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        # only get json/csv files
        files = [file for file in keys if os.path.splitext(file)[1] == "." + file_type]
        file_paths = [os.path.join("s3://" + bucket, file) for file in files]
        return file_paths


def extract_one_path(file_path, file_type, env):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    if prefix == "s3":
        file_paths = list_s3_file(file_url_splits[1], file_type, env)
    elif prefix == "hdfs":
        import pyarrow as pa
        fs = pa.hdfs.connect()
        if fs.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    else:
        if os.path.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    return file_paths


def get_node_ip():
    """
    This function is ported from ray to get the ip of the current node. In the settings where
    Ray is not involved, calling ray.services.get_node_ip_address would introduce Ray overhead.
    """
    import socket
    import errno
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet connection.
        s.connect(("8.8.8.8", 80))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()
    return node_ip_address
