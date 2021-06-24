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

AUTO_MODEL_SUPPORT_LIST = ["lstm", "tcn"]


class AutoModelFactory:
    @staticmethod
    def create_auto_model(name, search_space):
        name = name.lower()
        if name == "lstm":
            from .auto_lstm import AutoLSTM
            revised_search_space = search_space.copy()
            del revised_search_space["future_seq_len"]  # future_seq_len should always be 1
            return AutoLSTM(**revised_search_space)
        if name == "tcn":
            from .auto_tcn import AutoTCN
            return AutoTCN(**search_space)
        return NotImplementedError(f"{AUTO_MODEL_SUPPORT_LIST} are supported for auto model,\
                                    but get {name}.")
