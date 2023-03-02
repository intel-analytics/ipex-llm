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


# from multiprocessing import Process
from threading import Thread
from abc import abstractmethod
from . import TRIGGER_REG_NAME_PREFIX


class ConfigGenerator:
    '''
    Users could customize their own config generator by define a class
    inherited from ConfigGenerator.

    Example:
        >>> class MyConfigGenerator(ConfigGenerator):
        >>>     def __init__(self, ...):
        >>>         super().__init__()  # <-- This super class initialization must be called
        >>>         ...
        >>>     def genConfig(self, ...):  # <-- (recommeneded), generate a best config
        >>>         ...

    Users could also add some trigger decorator imported from `bigdl.chronos.aiops.trigger`
    More details about trigger please refer to correspond apis

    Example:
        >>> class MyConfigGenerator(ConfigGenerator):
        >>>     ...
        >>>     @triggerbyfile(...)
        >>>     def update_status(self, ...)  # <-- this will be exec once trigger activated
        >>>         ...
    '''
    def __init__(self):
        self.p_list = []
        for func in dir(self):
            if hasattr(getattr(self, func), "__name__"):
                if getattr(self, func).__name__.startswith(TRIGGER_REG_NAME_PREFIX):
                    p = Thread(target=getattr(self, func), args=(), daemon=True)
                    p.start()

    @abstractmethod
    def genConfig(self, *args, **kwargs):
        '''
        This function should be implemented for final config generation
        '''
        pass
