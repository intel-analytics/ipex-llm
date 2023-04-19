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

import torch


class CustomDict(dict):
    """Custom Dict class for class with dataclass, esp. for BaseOutput in diffusers
    """
    def __getattr__(self, key):
        if key in self.keys():
            return self.__getitem__(key)
        else:
            super().__getattr__(key)


class BasicMetaDataInfo(object):
    """Meta info class for basic type like int, float, str
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.need_reconstruct = False


class TensorMetaDataInfo(object):
    """Meta info class for torch.Tensor
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.shape = input_element.shape
        self.need_reconstruct = False


class DictMetaDataInfo(object):
    """Meta info class for dict
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        # normal dict or custom class
        random_key = list(input_element.keys())[0]
        if hasattr(input_element, random_key):
            # has attr, is custom class
            # fake a dataclass
            self.class_fn = CustomDict
        else:
            # normal dict
            self.class_fn = dict
        self.length = len(input_element)
        self.keys = list(input_element.keys())
        self.infos = []
        for k in self.keys:
            v = input_element[k]
            info = get_meta_info_from_input(v)
            if isinstance(info, ListMetaDataInfo):
                # list inside dict requires reconstruction
                info.need_reconstruct = True
            self.infos.append(info)
        self.need_reconstruct = True


class CustomClassMetaDataInfo(object):
    """Meta info class for custom class which is instance of dict
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.length = len(input_element)
        self.keys = list(input_element.keys())
        self.infos = []
        for k in self.keys:
            v = input_element[k]
            info = get_meta_info_from_input(v)
            if isinstance(info, ListMetaDataInfo):
                # list inside dict requires reconstruction
                info.need_reconstruct = True
            self.infos.append(info)
        self.need_reconstruct = True


class ListMetaDataInfo(object):
    """Meta info class for list output

    This is the main class for list output and its internal info instance can be
    of various types, for example:
    1. output is [Tensor, Tensor], then infos will contain two TensorMetaDataInfo instances.
    2. output is [Tensor, [Tensor, ...]], then infos will contain a TensorMetaDataInfo instance
       and a ListMetaDataInfo instance.
    3. output is [Tensor, {"sample: Tensor, ...}], then infos will contain a TensorMetaDataInfo
       instance and a DictMetaDataInfo instance.
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.length = len(input_element)
        self.need_reconstruct = False
        self.infos = []
        for ele in input_element:
            info = get_meta_info_from_input(ele)
            if isinstance(info, ListMetaDataInfo):
                # Nested list requires reconstruction
                info.need_reconstruct = True
            self.infos.append(info)
            self.need_reconstruct |= info.need_reconstruct


def get_meta_info_from_input(input_element):
    if isinstance(input_element, torch.Tensor):
        return TensorMetaDataInfo(input_element)
    elif isinstance(input_element, (tuple, list)):
        return ListMetaDataInfo(input_element)
    elif isinstance(input_element, dict):
        return DictMetaDataInfo(input_element)
    else:
        return BasicMetaDataInfo(input_element)


class MetaData(object):
    def __init__(self):
        pass

    @staticmethod
    def construct_matadata(output):
        return get_meta_info_from_input(output)

    @staticmethod
    def reconstruct_output(output, metadata):
        if not metadata.need_reconstruct:
            return output
        elif isinstance(metadata, DictMetaDataInfo):
            # Single dict
            if not isinstance(output, (tuple, list)):
                output = [output]
            new_output = metadata.class_fn()
            ind_out = 0
            for idx, k in enumerate(metadata.keys):
                meta = metadata.infos[idx]
                if not meta.need_reconstruct:
                    new_output[k] = output[ind_out]
                    ind_out += 1
                else:
                    new_output[k] = MetaData.reconstruct_output(
                        output[ind_out:], meta)
                    ind_out += meta.length
            return new_output
        else:
            # ListMetaDataInfo
            new_output = []
            ind_out = 0
            for index in range(metadata.length):
                meta = metadata.infos[index]
                if not meta.need_reconstruct:
                    new_output.append(output[ind_out])
                    ind_out += 1
                else:
                    # convert output based on current matadata, recursive call
                    new_output.append(MetaData.reconstruct_output(
                        output[ind_out:], meta))
                    ind_out += meta.length
            return new_output
