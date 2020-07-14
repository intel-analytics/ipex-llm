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

from setuptools import setup

TEMP_PATH = "bigdl/share"
bigdl_home = os.path.abspath(__file__ + "/../../")

try:
    exec(open('bigdl/version.py').read())
except IOError:
    print("Failed to load Bigdl version file for packaging. You must be in Bigdl's pyspark dir.")
    sys.exit(-1)

VERSION = __version__

building_error_msg = """
If you are packing python API from BigDL source, you must build BigDL first
and run sdist.
    To build BigDL with maven you can run:
      cd $BigDL_HOME
      ./make-dist.sh
    Building the source dist is done in the Python directory:
      cd pyspark
      python setup.py sdist
      pip install dist/*.tar.gz"""

def build_from_source():
    code_path = bigdl_home + "/pyspark/bigdl/util/common.py"
    print("Checking: %s to see if build from source" % code_path)
    if os.path.exists(code_path):
        return True
    return False


def init_env():
    if build_from_source():
        print("Start to build distributed package")
        print("HOME OF BIGDL: " + bigdl_home)
        dist_source = bigdl_home + "/dist"
        if not os.path.exists(dist_source):
            print(building_error_msg)
            sys.exit(-1)
        if os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
        copytree(dist_source, TEMP_PATH)
        copyfile(bigdl_home + "/pyspark/bigdl/nn/__init__.py", TEMP_PATH + "/__init__.py")
    else:
        print("Do nothing for release installation")

def get_bigdl_packages():
    bigdl_python_home = os.path.abspath(__file__)[:-8]
    bigdl_packages = ['bigdl.share']
    for dirpath, dirs, files in os.walk(bigdl_python_home + "bigdl"):
        package = dirpath.split(bigdl_python_home)[1].replace('/', '.')
        if "__pycache__" not in package:
            bigdl_packages.append(package)
    print "=========================bigdl packages========================="
    print "\n".join(bigdl_packages)
    print "================================================================"
    return bigdl_packages

def setup_package():
    metadata = dict(
        name='BigDL',
        version=VERSION,
        description='Distributed Deep Learning Library for Apache Spark',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/Bigdl',
        packages=get_bigdl_packages(),
        install_requires=['numpy>=1.7', 'pyspark==2.4.3', 'six>=1.10.0'],
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        package_data={"bigdl.share": ['bigdl/share/lib', 'bigdl/share/conf', 'bigdl/share/bin']},
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    try:
        init_env()
        setup_package()
    except Exception as e:
        raise e
    finally:
        if build_from_source() and os.path.exists(TEMP_PATH):
             rmtree(TEMP_PATH)

