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

import zoo.orca.automl.hp as hp


AUTO_MODEL_SUPPORT_LIST = ["lstm", "tcn", "seq2seq"]

AUTO_MODEL_DEFAULT_SEARCH_SPACE = {
    "lstm": {"minimal": {"hidden_dim": hp.grid_search([16, 32]),
                         "layer_num": hp.randint(1, 2),
                         "lr": hp.loguniform(0.001, 0.005),
                         "dropout": hp.uniform(0.1, 0.2)},
             "normal": {"hidden_dim": hp.grid_search([16, 32, 64]),
                        "layer_num": hp.grid_search([1, 2]),
                        "lr": hp.loguniform(0.0005, 0.01),
                        "dropout": hp.uniform(0, 0.2)},
             "large": {"hidden_dim": hp.grid_search([16, 32, 64, 128]),
                       "layer_num": hp.grid_search([1, 2, 3, 4]),
                       "lr": hp.loguniform(0.0005, 0.01),
                       "dropout": hp.uniform(0, 0.3)}},

    "tcn": {"minimal": {"hidden_units": hp.grid_search([16, 32]),
                        "levels": hp.randint(4, 6),
                        "kernel_size": 3,
                        "lr": hp.loguniform(0.001, 0.005),
                        "dropout": hp.uniform(0.1, 0.2)},
            "normal": {"hidden_units": hp.grid_search([16, 32, 48]),
                       "levels": hp.grid_search([6, 8]),
                       "kernel_size": hp.grid_search([3, 5]),
                       "lr": hp.loguniform(0.001, 0.01),
                       "dropout": hp.uniform(0, 0.2)},
            "large": {"hidden_units": hp.grid_search([16, 32, 48, 64]),
                      "levels": hp.grid_search([4, 5, 6, 7, 8]),
                      "kernel_size": hp.grid_search([3, 5, 7]),
                      "lr": hp.loguniform(0.0005, 0.015),
                      "dropout": hp.uniform(0, 0.25)}},

    "seq2seq": {"minimal": {"lr": hp.loguniform(0.001, 0.005),
                            "lstm_hidden_dim": hp.grid_search([16, 32]),
                            "lstm_layer_num": hp.randint(1, 2),
                            "dropout": hp.uniform(0, 0.3),
                            "teacher_forcing": False},
                "normal": {"lr": hp.loguniform(0.001, 0.005),
                           "lstm_hidden_dim": hp.grid_search([16, 32, 64]),
                           "lstm_layer_num": hp.grid_search([1, 2]),
                           "dropout": hp.uniform(0, 0.3),
                           "teacher_forcing": hp.grid_search([True, False])},
                "large": {"lr": hp.loguniform(0.0005, 0.005),
                          "lstm_hidden_dim": hp.grid_search([16, 32, 64, 128]),
                          "lstm_layer_num": hp.grid_search([1, 2, 4]),
                          "dropout": hp.uniform(0, 0.3),
                          "teacher_forcing": hp.grid_search([True, False])}}
}


class AutoModelFactory:
    @staticmethod
    def create_auto_model(name, search_space):
        name = name.lower()
        if name == "lstm":
            from .auto_lstm import AutoLSTM
            revised_search_space = search_space.copy()
            assert revised_search_space["future_seq_len"] == 1, \
                "future_seq_len should be set to 1 if you choose lstm model."
            del revised_search_space["future_seq_len"]  # future_seq_len should always be 1
            return AutoLSTM(**revised_search_space)
        if name == "tcn":
            from .auto_tcn import AutoTCN
            return AutoTCN(**search_space)
        if name == "seq2seq":
            from .auto_seq2seq import AutoSeq2Seq
            return AutoSeq2Seq(**search_space)
        return NotImplementedError(f"{AUTO_MODEL_SUPPORT_LIST} are supported for auto model,\
                                    but get {name}.")

    @staticmethod
    def get_default_search_space(model, computing_resource="normal"):
        '''
        This function should be called internally to get a default search_space experimentally.

        :param model: model name, only tcn, lstm and seq2seq are supported
        :param mode: one of "minimal", "normal", "large"
        '''
        model = model.lower()
        if model in AUTO_MODEL_SUPPORT_LIST:
            return AUTO_MODEL_DEFAULT_SEARCH_SPACE[model][computing_resource]
        return NotImplementedError(f"{AUTO_MODEL_SUPPORT_LIST} are supported for auto model,\
                                        but get {model}.")
