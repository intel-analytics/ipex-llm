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
from shutil import copytree, rmtree
from setuptools import setup

bigdl_home = os.path.abspath(__file__ + "/../../../..")
VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()
SCRIPTS_TARGET = os.path.join(bigdl_home, "scala/serving/scripts/")
TMP_PATH = "bigdl/conf"
if os.path.exists(TMP_PATH):
    rmtree(TMP_PATH)
copytree(SCRIPTS_TARGET, TMP_PATH)


def setup_package():
    script_names = [f for f in os.listdir(SCRIPTS_TARGET) if
                    os.path.isfile(os.path.join(SCRIPTS_TARGET, f))]
    scripts = list(map(lambda script: os.path.join(
        SCRIPTS_TARGET, script), script_names))

    metadata = dict(
        name='bigdl-serving',
        version=VERSION,
        description='A unified Data Analytics and AI platform for distributed TensorFlow, Keras, '
                    'PyTorch, Apache Spark/Flink and Ray',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=['bigdl.serving', 'bigdl.conf'],
        package_dir={'bigdl.serving': '../../serving/'},
        package_data={"bigdl.conf": ['config.yaml']},
        include_package_data=False,
        scripts=scripts,
        install_requires=['redis', 'pyyaml', 'httpx', 'pyarrow', 'opencv-python'],
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    try:
        setup_package()
    except Exception as e:
        raise e
