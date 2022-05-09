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
from bigdl.dllib.utils.common import Sample as BSample, JTensor as BJTensor,\
    JavaCreator, _get_gateway, _py2java, _java2py
import numpy as np
import os
import tempfile
import uuid
import functools
import glob

from urllib.parse import urlparse
from bigdl.dllib.utils.log4Error import *


def convert_to_safe_path(input_path, follow_symlinks=True):
    # resolves symbolic links
    if follow_symlinks:
        return os.path.realpath(input_path)
    # covert to abs path
    return os.path.abspath(input_path)


def to_list_of_numpy(elements):
    if isinstance(elements, np.ndarray):
        return [elements]
    elif np.isscalar(elements):
        return [np.array(elements)]
    elif not isinstance(elements, list):
        invalidInputError(False, "Wrong type: %s" % type(elements))

    results = []
    for element in elements:
        if np.isscalar(element):
            results.append(np.array(element))
        elif isinstance(element, np.ndarray):
            results.append(element)
        else:
            invalidInputError(False, "Wrong type: %s" % type(element))

    return results


def get_file_list(path, recursive=False):
    return callZooFunc("float", "listPaths", path, recursive)


def exists(path):
    return callZooFunc("float", "exists", path)


def mkdirs(path):
    callZooFunc("float", "mkdirs", path)


def is_local_path(path):
    parse_result = urlparse(path)
    return len(parse_result.scheme.lower()) == 0 or parse_result.scheme.lower() == "file"


def append_suffix(prefix, path):
    # append suffix
    splits = path.split(".")
    if len(splits) > 0:
        file_name = prefix + "." + splits[-1]
    else:
        file_name = prefix

    return file_name


def enable_multi_fs_save(save_func):

    @functools.wraps(save_func)
    def save_mult_fs(obj, path, *args, **kwargs):
        if is_local_path(path):
            return save_func(obj, path, *args, **kwargs)
        else:
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path)
            temp_path = os.path.join(tempfile.gettempdir(), file_name)

            try:
                result = save_func(obj, temp_path, *args, **kwargs)
                if "overwrite" in kwargs:
                    put_local_file_to_remote(temp_path, path, over_write=kwargs['overwrite'])
                else:
                    put_local_file_to_remote(temp_path, path)
            finally:
                os.remove(temp_path)
            return result

    return save_mult_fs


def enable_multi_fs_load_static(load_func):
    @functools.wraps(load_func)
    def multi_fs_load(path, *args, **kwargs):
        if is_local_path(path):
            return load_func(path, *args, **kwargs)
        else:
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path)
            temp_path = os.path.join(tempfile.gettempdir(), file_name)
            get_remote_file_to_local(path, temp_path)
            try:
                return load_func(temp_path, *args, **kwargs)
            finally:
                os.remove(temp_path)

    return multi_fs_load


def enable_multi_fs_load(load_func):

    @functools.wraps(load_func)
    def multi_fs_load(obj, path, *args, **kwargs):
        if is_local_path(path):
            return load_func(obj, path, *args, **kwargs)
        else:
            file_name = str(uuid.uuid1())
            file_name = append_suffix(file_name, path)
            temp_path = os.path.join(tempfile.gettempdir(), file_name)
            get_remote_file_to_local(path, temp_path)
            try:
                return load_func(obj, temp_path, *args, **kwargs)
            finally:
                os.remove(temp_path)

    return multi_fs_load


def get_remote_file_to_local(remote_path, local_path, over_write=False):
    callZooFunc("float", "getRemoteFileToLocal", remote_path, local_path, over_write)


def get_remote_dir_to_local(remote_dir, local_dir):
    # get remote file lists
    file_list = get_file_list(remote_dir)
    # get remote files to local
    [get_remote_file_to_local(file, os.path.join(local_dir, os.path.basename(file)))
     for file in file_list]


def get_remote_files_with_prefix_to_local(remote_path_prefix, local_dir):
    remote_dir = os.path.dirname(remote_path_prefix)
    prefix = os.path.basename(remote_path_prefix)
    # get remote file lists
    file_list = get_file_list(remote_dir)
    file_list = [file for file in file_list if os.path.basename(file).startswith(prefix)]
    # get remote files to local
    [get_remote_file_to_local(file, os.path.join(local_dir, os.path.basename(file)))
     for file in file_list]


def get_remote_dir_tree_to_local(remote_dir, local_dir):
    if os.path.exists(local_dir):
        os.makedirs(local_dir)
    # get remote file lists
    file_list = get_file_list(remote_dir, recursive=True)
    for file in file_list:
        local_subdir = os.path.join(local_dir, os.path.dirname(file)[len(remote_dir)+1:])
        filename = os.path.basename(file)
        if not os.path.exists(local_subdir):
            os.makedirs(local_subdir)
        get_remote_file_to_local(file, os.path.join(local_subdir, filename))


def put_local_file_to_remote(local_path, remote_path, over_write=False):
    callZooFunc("float", "putLocalFileToRemote", local_path, remote_path, over_write)


def put_local_files_with_prefix_to_remote(local_path_prefix, remote_dir, over_write=False):
    # get local file lists
    file_list = glob.glob(local_path_prefix + "*")
    # get remote files to local
    [put_local_file_to_remote(file, os.path.join(remote_dir, os.path.basename(file)),
                              over_write=over_write)
     for file in file_list]


def put_local_dir_tree_to_remote(local_dir, remote_dir, over_write=False):
    if not exists(remote_dir):
        mkdirs(remote_dir)
    for dirpath, dirnames, filenames in os.walk(local_dir):
        for d in dirnames:
            remote_subdir = os.path.join(remote_dir, dirpath[len(local_dir)+1:], d)
            if not exists(remote_subdir):
                mkdirs(remote_subdir)
        for f in filenames:
            remote_file = os.path.join(remote_dir, dirpath[len(local_dir)+1:], f)
            put_local_file_to_remote(os.path.join(dirpath, f), remote_file, over_write=over_write)


def set_core_number(num):
    callZooFunc("float", "setCoreNumber", num)


def callZooFunc(bigdl_type, name, *args):
    """ Call API in PythonBigDL """
    gateway = _get_gateway()
    args = [_py2java(gateway, a) for a in args]
    error = Exception("Cannot find function: %s" % name)
    for jinvoker in JavaCreator.instance(bigdl_type, gateway).value:
        # hasattr(jinvoker, name) always return true here,
        # so you need to invoke the method to check if it exist or not
        try:
            api = getattr(jinvoker, name)
            java_result = api(*args)
            result = _java2py(gateway, java_result)
        except Exception as e:
            error = e
            if not ("does not exist" in str(e)
                    and "Method {}".format(name) in str(e)):
                invalidOperationError(False, str(e), cause=e)
        else:
            return result
    invalidOperationError(False, str(error), cause=error)


class JTensor(BJTensor):

    def __init__(self, storage, shape, bigdl_type="float", indices=None):
        super(JTensor, self).__init__(storage, shape, bigdl_type, indices)

    @classmethod
    def from_ndarray(cls, a_ndarray, bigdl_type="float"):
        """
        Convert a ndarray to a DenseTensor which would be used in Java side.
        """
        if a_ndarray is None:
            return None
        invalidInputError(isinstance(a_ndarray, np.ndarray),
                          "input should be a np.ndarray, not %s" % type(a_ndarray))
        return cls(a_ndarray,
                   a_ndarray.shape,
                   bigdl_type)


class Sample(BSample):

    def __init__(self, features, labels, bigdl_type="float"):
        super(Sample, self).__init__(features, labels, bigdl_type)

    @classmethod
    def from_ndarray(cls, features, labels, bigdl_type="float"):
        features = to_list_of_numpy(features)
        labels = to_list_of_numpy(labels)
        return cls(
            features=[JTensor(feature, feature.shape) for feature in features],
            labels=[JTensor(label, label.shape) for label in labels],
            bigdl_type=bigdl_type)
