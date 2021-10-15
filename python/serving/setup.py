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
import sys
from shutil import copyfile, copytree, rmtree
import fnmatch
from setuptools import setup

long_description = '''
Analytics Zoo: A unified Data Analytics and AI platform for distributed TensorFlow,
 Keras, PyTorch, Apache Spark/Flink and Ray.

You may want to develop your AI solutions using Analytics Zoo if:

- You want to easily prototype the entire end-to-end pipeline that applies AI models
 (e.g., TensorFlow, Keras, PyTorch, BigDL, OpenVINO, etc.) to production big data.
- You want to transparently scale your AI applications from a laptop to large clusters with "zero"
 code changes.
- You want to deploy your AI pipelines to existing YARN or K8S clusters *WITHOUT* any modifications
 to the clusters.
- You want to automate the process of applying machine learning (such as feature engineering,
 hyperparameter tuning, model selection and distributed inference).

Find instructions to install analytics-zoo via pip, please visit our documentation page:
 https://analytics-zoo.github.io/master/#PythonUserGuide/install/

For source code and more information, please visit our GitHub page:
 https://github.com/intel-analytics/analytics-zoo
'''

bigdl_home = os.path.abspath(__file__ + "/../../../")
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
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='Analytics Zoo Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/bigdl',
        packages=['bigdl.serving', 'bigdl.conf'],
        package_dir={'bigdl.serving': '../serving/'},
        package_data={"bigdl.conf": ['config.yaml']},
        include_package_data=False,
        scripts=scripts,
        install_requires=['redis', 'pyyaml', 'httpx', 'pyarrow', 'opencv-python'],
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    try:
        setup_package()
    except Exception as e:
        raise e
