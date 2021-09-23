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

IDENTIFIER_LEN = 27


def process(command, fail_fast=False, timeout=120):
    import subprocess
    pro = subprocess.Popen(
        command,
        shell=True,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    out, err = pro.communicate(timeout=timeout)
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    print(out)
    print(err)
    errorcode = pro.returncode
    if errorcode != 0:
        if fail_fast:
            raise Exception(err)
        print(err)
    else:
        print(out)


def get_remote_list(dir_in):
    args = "hdfs dfs -ls " + dir_in + " | awk '{print $8}'"
    s_output, _ = process(args)

    all_dart_dirs = s_output.split()
    names = []
    for filename in all_dart_dirs:
        filename = filename.decode()
        name_list = filename.split('/')
        names.append(name_list[-1])
    return names


def put_ckpt_hdfs(remote_dir, ckpt_name):
    """
    Upload checkpoint file with name of ckpt_name to the hdfs directory
    {remote_dir}/{ray_checkpoint_dir}[-IDENTIFIER_LEN:].

    Note that ray_checkpoint_dir is like train_func_0_{config}_{time}_{tmp},
    with a max identifier length of 130. However, if there is a list("[]") in config and is
    truncated with half "[" remained, then the path name can't be identified by hadoop command.
    Therefore we use the last IDENTIFIER_LEN=27 of ray_checkpoint_dir as remote_ckpt_basename,
    with a format of {time}_{tmp}, in order to avoid misinterpretation.
    """

    local_ckpt_dir = os.path.abspath(".")
    remote_ckpt_basename = os.path.basename(local_ckpt_dir)[-IDENTIFIER_LEN:]
    remote_ckpt_dir = os.path.join(remote_dir, remote_ckpt_basename)
    if remote_ckpt_basename not in get_remote_list(remote_dir):
        cmd = f"hadoop fs -mkdir {remote_ckpt_dir};" \
              f" hadoop fs -put -f {ckpt_name} {remote_ckpt_dir}"
    else:
        cmd = f"hadoop fs -put -f {ckpt_name} {remote_ckpt_dir}"
    process(cmd)


def get_ckpt_hdfs(remote_dir, local_ckpt):
    """
    Get checkpoint file from hdfs as local_ckpt
    Remote checkpoint dir is {remote_dir}/{ray_checkpoint_dir}[-IDENTIFIER_LEN:].
    """
    ckpt_name = os.path.basename(local_ckpt)
    local_ckpt_dir = os.path.dirname(local_ckpt)
    remote_ckpt_basename = os.path.basename(local_ckpt_dir)[-IDENTIFIER_LEN:]
    remote_ckpt = os.path.join(remote_dir, remote_ckpt_basename, ckpt_name)

    cmd = "hadoop fs -get {} {}".format(remote_ckpt, local_ckpt_dir)
    process(cmd)
