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

TEMP_PATH = "bigdl/share/dllib"
dllib_src_path = os.path.abspath(__file__ + "/..")

try:
    exec(open(dllib_src_path + "/bigdl/dllib/version.py").read())
except IOError:
    print("Failed to load bigdl-dllib version file for packaging. "
          "You need to run the release script instead.")
    sys.exit(-1)

VERSION = __version__  # noqa

building_error_msg = """
If you are packing python API from BigDL source, you should use the release script:
    cd $BigDL_HOME/python/dllib/dev/release
    ./release.sh platform version quick_build upload mvn_parameters(if any)
After the build:
    cd $BigDL_HOME/python/dllib/src/dist
    pip install *.tar.gz
"""


if os.path.exists(dllib_src_path + "/bigdl_dllib_spark3.egg-info"):
    build_from_source = False
else:
    build_from_source = True


def init_env():
    if build_from_source:
        print("Start to build distributed package")
        bigdl_home = os.path.abspath(dllib_src_path + "/../../../")
        dist_source = bigdl_home + "/dist"
        if not os.path.exists(dist_source):
            print(building_error_msg)
            sys.exit(-1)
        if os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
        copytree(dist_source, TEMP_PATH)
        copyfile(dllib_src_path + "/bigdl/dllib/nn/__init__.py",
                 TEMP_PATH + "/__init__.py")
    else:
        print("Do nothing for release installation")


def get_bigdl_packages():
    bigdl_packages = ['bigdl.share.dllib']
    for dirpath, dirs, files in os.walk(dllib_src_path + "/bigdl"):
        package = dirpath.split(dllib_src_path + "/")[1].replace('/', '.')
        if "__pycache__" not in package:
            bigdl_packages.append(package)
    print("=========================bigdl packages=========================")
    print("\n".join(bigdl_packages))
    print("================================================================")
    return bigdl_packages


def setup_package():
    SCRIPTS_TARGET = "bigdl/scripts/"
    script_names = ["pyspark-with-bigdl", "spark-submit-with-bigdl"]
    scripts = list(map(lambda script: os.path.join(
        SCRIPTS_TARGET, script), script_names))
    copyfile(dllib_src_path + "/bigdl/dllib/nn/__init__.py",
             SCRIPTS_TARGET + "__init__.py")
    metadata = dict(
        name='bigdl-dllib-spark3',
        version=VERSION,
        description='Distributed Deep Learning Library for Apache Spark',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=get_bigdl_packages(),
        scripts=scripts,
        install_requires=[
            'numpy>=1.19.5', 'pyspark==3.1.2', 'conda-pack==0.3.1',
            'six>=1.10.0', 'bigdl-core==2.1.0b20220321'],
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        package_data={"bigdl.share.dllib": ['lib/bigdl-dllib*.jar', 'conf/*',
                                            'bin/standalone/*', 'bin/standalone/sbin/*'],
                      "bigdl.scripts": script_names},
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
        init_env()
        setup_package()
    except Exception as e:
        raise e  # noqa
    finally:
        if build_from_source and os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
