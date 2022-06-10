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
import subprocess

import shutil
import glob
import os
from bigdl.serving.log4Error import invalidInputError


class ClusterServing:
    def __init__(self):
        self.conf_path = os.path.abspath(
            __file__ + "/../../conf/config.yaml")
        self.copy_config()

    def copy_config(self):
        if os.path.exists("config.yaml"):
            return
        print("Trying to find config file in ", self.conf_path)
        if not os.path.exists(self.conf_path):
            print('WARNING: Config file does not exist in your pip directory,'
                  'are you sure that you install serving by pip?')
            build_conf_path = glob.glob(os.path.abspath(
                __file__ + "/../../../../scripts/cluster-serving/config.yaml"))
            prebuilt_conf_path = glob.glob(os.path.abspath(
                __file__ + "/../../../../../bin/cluster-serving/config.yaml"))
            conf_paths = build_conf_path + prebuilt_conf_path

            invalidInputError(len(conf_paths) > 0, "No config file is found")
            self.conf_path = conf_paths[0]
            print("config path is found at ", self.conf_path)

            if not os.path.exists(self.conf_path):
                invalidInputError(False,
                                  "Can not find your config file.")
        else:
            print('Config file found in pip package, copying...')
        try:
            shutil.copyfile(self.conf_path, 'config.yaml')
            print('Config file ready.')
        except Exception as e:
            print(e)
            print("WARNING: An initialized config file already exists.")

        subprocess.Popen(['chmod', 'a+x', self.conf_path])
