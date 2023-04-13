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
from ..common import invalidInputError


class BaiscMetaDataInfo(object):
    """Meta info class for basic type like int, float, str
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.reconstruct = False


class TensorMetaDataInfo(object):
    """Meta info class for torch.Tensor
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.shape = input_element.shape
        self.reconstruct = False


class DictMetaDataInfo(object):
    """Meta info class for dict
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.length = len(input_element)
        self.keys = list(input_element.keys())
        self.infos = []
        for k in self.keys:
            v = input_element[k]
            info = get_meta_info_from_input(v)
            self.infos.append(info)
        self.reconstruct = True


class ListMetaDataInfo(object):
    """Meta info class for list
    """
    def __init__(self, input_element):
        self.type = type(input_element)
        self.length = len(input_element)
        self.reconstruct = False
        self.infos = []
        for ele in input_element:
            info = get_meta_info_from_input(ele)
            self.infos.append(info)
            self.reconstruct |= info.reconstruct


def get_meta_info_from_input(input_element):
    if isinstance(input_element, torch.Tensor):
        return TensorMetaDataInfo(input_element)
    elif isinstance(input_element, (tuple, list)):
        return ListMetaDataInfo(input_element)
    elif isinstance(input_element, dict):
        return DictMetaDataInfo(input_element)
    else:
        return BaiscMetaDataInfo(input_element)


class MetaData(object):
    def __init__(self):
        pass

    @staticmethod
    def construct_matadata(output):
        return get_meta_info_from_input(output)

    @staticmethod
    def reconstruct_output(output, metadata):
        # TODO: support more cases
        if not metadata.reconstruct:
            return output
        elif isinstance(metadata, DictMetaDataInfo):
            # single dict
            if not isinstance(output, (tuple, list)):
                output = [output]
            new_output = {}
            elem_idx = 0
            for idx, k in enumerate(metadata.keys):
                meta = metadata.infos[idx]
                if meta not in (DictMetaDataInfo, ListMetaDataInfo):
                    new_output[k] = output[elem_idx]
                    elem_idx += 1
                else:
                    # TODO: dict contain list
                    pass
            return new_output
        else:
            # ListMetaDataInfo
            new_output = []
            ind_out = 0
            ind_meta = 0
            for index in range(metadata.length):
                meta = metadata.infos[index]
                if not meta.reconstruct:
                    new_output.append(output[ind_out])
                    ind_out += 1
                else:
                    # convert output based on current matadata, recursive call
                    new_output.append(MetaData.reconstruct_output(
                        output[ind_out:], meta))
                    ind_out += meta.length
            return new_output
