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

from zoo import ZooContext


class OrcaContextMeta(type):

    _pandas_read_backend = "spark"
    __eager_mode = True
    _serialize_data_creation = False

    @property
    def log_output(cls):
        """
        Whether to redirect Spark driver JVM's stdout and stderr to the current
        python process. This is useful when running Analytics Zoo in jupyter notebook.
        Default to be False. Needs to be set before initializing SparkContext.
        """
        return ZooContext.log_output

    @log_output.setter
    def log_output(cls, value):
        ZooContext.log_output = value

    @property
    def pandas_read_backend(cls):
        """
        The backend for reading csv/json files. Either "spark" or "pandas".
        spark backend would call spark.read and pandas backend would call pandas.read.
        Default to be "spark".
        """
        return cls._pandas_read_backend

    @pandas_read_backend.setter
    def pandas_read_backend(cls, value):
        value = value.lower()
        assert value == "spark" or value == "pandas", \
            "pandas_read_backend must be either spark or pandas"
        cls._pandas_read_backend = value

    @property
    def _eager_mode(cls):
        """
        Whether to compute eagerly for SparkXShards.
        Default to be True.
        """
        return cls.__eager_mode

    @_eager_mode.setter
    def _eager_mode(cls, value):
        assert isinstance(value, bool), "_eager_mode should either be True or False"
        cls.__eager_mode = value

    @property
    def serialize_data_creation(cls):
        """
        Whether add a file lock to the data loading process for PyTorch Horovod training.
        This would be useful when you run multiple workers on a single node to download data
        to the same destination.
        Default to be False.
        """
        return cls._serialize_data_creation

    @serialize_data_creation.setter
    def serialize_data_creation(cls, value):
        assert isinstance(value, bool), "serialize_data_creation should either be True or False"
        cls._serialize_data_creation = value


class OrcaContext(metaclass=OrcaContextMeta):
    pass
