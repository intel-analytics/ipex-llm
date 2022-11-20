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

from abc import ABCMeta, abstractmethod
import io
import torch
from bigdl.orca.learn.pytorch.utils import get_filesystem
from bigdl.dllib.utils.log4Error import invalidInputError


class ModelIO(metaclass=ABCMeta):
    @abstractmethod
    def get_state_dict(self):
        """Returns the state of the runner."""
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """Sets the state of the model."""
        pass

    @abstractmethod
    def save_checkpoint(self, filepath, save_weights_only=False):
        """Save checkpoint."""
        pass

    @abstractmethod
    def remove_checkpoint(self, filepath):
        """Remove checkpoint"""
        pass

    def get_state_stream(self):
        """Returns a bytes object for the state dict."""
        state_dict = self.get_state_dict()
        state_stream = ModelIO._state_dict2stream(state_dict)
        return state_stream

    def load_state_stream(self, byte_obj):
        """Loads a bytes object the training state dict."""
        state_dict = ModelIO._state_stream2dict(byte_obj)
        return self.load_state_dict(state_dict)

    def load_checkpoint(self, filepath):
        fs = get_filesystem(filepath)
        if not fs.exists(filepath):
            invalidInputError(False,
                              f"Checkpoint at {filepath} not found. Aborting training.")
        with fs.open(filepath, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    @staticmethod
    def _state_dict2stream(state_dict):
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        return _buffer.getvalue()

    @staticmethod
    def _state_stream2dict(byte_obj):
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer)
        return state_dict
