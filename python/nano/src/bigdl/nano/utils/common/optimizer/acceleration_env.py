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

import bigdl
from .optimizer import register_suggestion

_whole_acceleration_env = ['tcmalloc', 'jemalloc', 'openmp', 'perf']


class AccelerationEnv(object):
    __slot__ = _whole_acceleration_env
    _CONDA_DIR = os.environ.get('CONDA_PREFIX', None)
    _NANO_DIR = os.path.join(os.path.dirname(bigdl.__file__), 'nano')

    def __init__(self, **kwargs):
        """
        initialize optimization env
        """
        self.openmp = None
        self.jemalloc = None
        self.tcmalloc = None
        self.perf = None
        for option in _whole_acceleration_env:
            setattr(self, option, kwargs.get(option, False))

    def get_malloc_lib(self):
        if self.tcmalloc:
            return "tcmalloc"
        if self.jemalloc:
            return "jemalloc"
        return None

    def get_omp_lib(self):
        if self.openmp and self.perf:
            return "openmp_perf"
        elif self.openmp:
            return "openmp"
        else:
            return None

    def get_env_dict(self):
        tmp_env_dict = {}

        # set allocator env var
        tmp_malloc_lib = self.get_malloc_lib()
        if tmp_malloc_lib == 'jemalloc':
            tmp_env_dict['LD_PRELOAD'] = os.path.join(
                AccelerationEnv._NANO_DIR, 'libs/libjemalloc.so')
            tmp_env_dict['MALLOC_CONF'] = 'oversize_threshold:1,background_thread:false,' \
                                          'metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1'
        elif tmp_malloc_lib:
            tmp_env_dict['LD_PRELOAD'] = os.path.join(
                AccelerationEnv._NANO_DIR, 'libs/libtcmalloc.so')
            tmp_env_dict['MALLOC_CONF'] = ''
        else:
            tmp_env_dict['LD_PRELOAD'] = ''
            tmp_env_dict['MALLOC_CONF'] = ''

        # set omp env var
        omp_lib_path = ''
        if AccelerationEnv._CONDA_DIR:
            if os.path.exists(os.path.join(AccelerationEnv._CONDA_DIR, '../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._CONDA_DIR, '../lib/libiomp5.so')
            elif os.path.exists(os.path.join(AccelerationEnv._CONDA_DIR,
                                             '../../../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._CONDA_DIR,
                                            '../../../lib/libiomp5.so')
            else:
                register_suggestion(f"No OpenMP library found in {AccelerationEnv._CONDA_DIR}."
                                    f"You can install OpenMP by "
                                    f"'conda install -c anaconda intel-openmp'")
        elif AccelerationEnv._NANO_DIR:
            if os.path.exists(os.path.join(AccelerationEnv._NANO_DIR,
                                           '../../../libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._NANO_DIR,
                                            '../../../libiomp5.so')
            elif os.path.exists(os.path.join(AccelerationEnv._NANO_DIR,
                                             '../../../../../../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._NANO_DIR,
                                            '../../../../../../lib/libiomp5.so')
            else:
                register_suggestion(f"No OpenMP library found in {AccelerationEnv._NANO_DIR}."
                                    f"You can install OpenMP by "
                                    f"'pip install intel-openmp'")

        tmp_omp_lib = self.get_omp_lib()
        if tmp_omp_lib == 'openmp_perf':
            tmp_env_dict['LD_PRELOAD'] = tmp_env_dict['LD_PRELOAD'] + ' ' + omp_lib_path
            tmp_env_dict['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
            tmp_env_dict['KMP_BLOCKTIME'] = '1'
        elif tmp_omp_lib == 'openmp':
            tmp_env_dict['LD_PRELOAD'] = tmp_env_dict['LD_PRELOAD'] + ' ' + omp_lib_path
            tmp_env_dict['KMP_AFFINITY'] = 'granularity=fine,none'
            tmp_env_dict['KMP_BLOCKTIME'] = '1'
        else:
            tmp_env_dict['KMP_AFFINITY'] = ''
            tmp_env_dict['KMP_BLOCKTIME'] = ''
        return tmp_env_dict
