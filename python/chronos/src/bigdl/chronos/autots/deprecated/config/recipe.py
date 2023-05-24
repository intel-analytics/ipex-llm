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

from bigdl.chronos.autots.deprecated.config.base import Recipe
from bigdl.orca.automl import hp
from bigdl.chronos.utils import deprecated


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class SmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        '''
        __init__()
        '''
        super(self.__class__, self).__init__()

    def search_space(self):
        return {
            "model": "LSTM",
            "lstm_1_units": hp.choice([32, 64]),
            "dropout_1": hp.uniform(0.2, 0.5),
            "lstm_2_units": hp.choice([32, 64]),
            "dropout_2": hp.uniform(0.2, 0.5),
            "lr": 0.001,
            "batch_size": 1024,
            "epochs": 1,
            "past_seq_len": 2,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class MTNetSmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        '''
        __init__()
        '''
        super(self.__class__, self).__init__()

    def search_space(self):
        return {
            "model": "MTNet",
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 1,
            "cnn_dropout": 0.2,
            "rnn_dropout": 0.2,
            "time_step": hp.choice([3, 4]),
            "cnn_height": 2,
            "long_num": hp.choice([3, 4]),
            "ar_size": hp.choice([2, 3]),
            "past_seq_len": hp.sample_from(lambda spec:
                                           (spec.config.long_num + 1) * spec.config.time_step),
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class TCNSmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        '''
        __init__()
        '''
        super(self.__class__, self).__init__()

    def search_space(self):
        return {
            "lr": 0.001,
            "batch_size": 16,
            "nhid": 8,
            "levels": 8,
            "kernel_size": 3,
            "dropout": 0.1
        }


class PastSeqParamHandler(object):
    """
    Utility to handle PastSeq Param
    """

    def __init__(self):
        pass

    @staticmethod
    @deprecated('Please use `bigdl.orca.automl.hp` instead.')
    def get_past_seq_config(look_back):
        """
        get_past_seq_config(look_back)
        Generate pass sequence config based on look_back.

        :param look_back: look_back configuration
        :return: search configuration for past sequence
        """
        from bigdl.nano.utils.common import invalidInputError
        if isinstance(
            look_back,
            tuple) and len(look_back) == 2 and isinstance(
                look_back[0],
                int) and isinstance(
                look_back[1],
                int):
            if look_back[1] < 2:
                invalidInputError(False,
                                  "The max look back value should be at least 2")
            if look_back[0] < 2:
                print(
                    "The input min look back value is smaller than 2. "
                    "We sample from range (2, {}) instead.".format(
                        look_back[1]))
            past_seq_config = hp.randint(look_back[0], look_back[1] + 1)
        elif isinstance(look_back, int):
            if look_back < 2:
                invalidInputError(False,
                                  "look back value should not be smaller than 2. "
                                  "Current value is ", look_back)
            past_seq_config = look_back
        else:
            invalidInputError(False,
                              "look back is {}.\n "
                              "look_back should be either a tuple with 2 int values:"
                              " (min_len, max_len) or a single int".format(look_back))
        return past_seq_config


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class GridRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search.
    """

    def __init__(
            self,
            num_rand_samples=1,
            look_back=2,
            epochs=5,
            training_iteration=10):
        """
        __init__()
        Constructor.
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)
        self.epochs = epochs

    def search_space(self):
        return {
            # -------- model selection TODO add MTNet
            "model": hp.choice(["LSTM", "Seq2seq"]),

            # --------- Vanilla LSTM model parameters
            "lstm_1_units": hp.grid_search([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": hp.grid_search([16, 32]),
            "dropout_2": hp.uniform(0.2, 0.5),

            # ----------- Seq2Seq model parameters
            "latent_dim": hp.grid_search([32, 64]),
            "dropout": hp.uniform(0.2, 0.5),

            # ----------- optimization parameters
            "lr": hp.uniform(0.001, 0.01),
            "batch_size": hp.choice([32, 64]),
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class LSTMSeq2SeqRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search, only for Seq2SeqPytorch.
    Note: This recipe is specifically designed for third-party model searching, rather
    than TimeSequencePredictor.
    """

    def __init__(
            self,
            input_feature_num,
            output_feature_num,
            future_seq_len,
            num_rand_samples=1,
            epochs=1,
            training_iteration=20,
            batch_size=[128, 256, 512],
            lr=(0.001, 0.01),
            lstm_hidden_dim=[64, 128],
            lstm_layer_num=[1, 2, 3, 4],
            dropout=(0, 0.25),
            teacher_forcing=[True, False]):
        """
        __init__()
        Constructor.
        set the param to a list for grid search.
        set the param to a tuple with length = 2 for random search.

        :param input_feature_num: (int) no. of input feature
        :param output_feature_num: (int) no. of ouput feature
        :param future_seq_len: (int) no. of steps to be predicted (i.e. horizon)
        :param num_rand_samples: (int) number of hyper-param configurations sampled randomly
        :param epochs: (int) no. of epochs to train in each iteration
        :param training_iteration: (int) no. of iterations for training (n epochs) in trials
        :param batch_size: (tuple|list) grid search candidates for batch size
        :param lr: (tuple|list) learning rate
        :param lstm_hidden_dim: (tuple|list) lstm hidden dim for both encoder and decoder
        :param lstm_layer_num: (tuple|list) no. of lstm layer for both encoder and decoder
        :param dropout: (tuple|list) dropout for lstm layer
        :param teacher_forcing: (list) if to use teacher forcing machanism during training
        """
        super(self.__class__, self).__init__()
        # -- runtime params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- optimization params
        self.lr = self._gen_sample_func(lr, "lr")
        self.batch_size = self._gen_sample_func(batch_size, "batch_size")
        self.epochs = epochs

        # -- model params
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.future_seq_len = future_seq_len
        self.lstm_hidden_dim = self._gen_sample_func(
            lstm_hidden_dim, "lstm_hidden_dim")
        self.lstm_layer_num = self._gen_sample_func(
            lstm_layer_num, "lstm_layer_num")
        self.dropout = self._gen_sample_func(dropout, "dropout")
        self.teacher_forcing = self._gen_sample_func(
            teacher_forcing, "teacher_forcing")

    def _gen_sample_func(self, ranges, param_name):
        from bigdl.nano.utils.common import invalidInputError
        if isinstance(ranges, tuple):
            invalidInputError(len(ranges) == 2,
                              f"length of tuple {param_name} should be"
                              f" 2 while get {len(ranges)} instead.")
            invalidInputError(param_name != "teacher_forcing",
                              f"type of {param_name} can only be a list while get a tuple")
            if param_name in ["lr"]:
                return hp.loguniform(lower=ranges[0], upper=ranges[1])
            if param_name in ["lstm_hidden_dim",
                              "lstm_layer_num", "batch_size"]:
                return hp.randint(lower=ranges[0], upper=ranges[1])
            if param_name in ["dropout"]:
                return hp.uniform(lower=ranges[0], upper=ranges[1])
        if isinstance(ranges, list):
            return hp.grid_search(ranges)
            invalidInputError(False, f"{param_name} should be either a list or a tuple.")

    def search_space(self):
        return {
            # ----------- data parameters
            "input_feature_num": self.input_feature_num,
            "output_feature_num": self.output_feature_num,
            "future_seq_len": self.future_seq_len,
            # ----------- optimization parameters
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            # ----------- model parameters
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_layer_num": self.lstm_layer_num,
            "dropout": self.dropout,
            "teacher_forcing": self.teacher_forcing,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class LSTMGridRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search, only for LSTM.
    """

    def __init__(
            self,
            num_rand_samples=1,
            epochs=5,
            training_iteration=10,
            look_back=2,
            lstm_1_units=[16, 32, 64, 128],
            lstm_2_units=[16, 32, 64],
            batch_size=[32, 64]):
        """
        __init__()
        Constructor.

        :param lstm_1_units: random search candidates for num of lstm_1_units
        :param lstm_2_units: grid search candidates for num of lstm_1_units
        :param batch_size: grid search candidates for batch size
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        # -- runtime params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- model params
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)
        self.lstm_1_units_config = hp.choice(lstm_1_units)
        self.lstm_2_units_config = hp.grid_search(lstm_2_units)
        self.dropout_2_config = hp.uniform(0.2, 0.5)

        # -- optimization params
        self.lr = hp.uniform(0.001, 0.01)
        self.batch_size = hp.grid_search(batch_size)
        self.epochs = epochs

    def search_space(self):
        return {
            "model": "LSTM",

            # --------- Vanilla LSTM model parameters
            "lstm_1_units": self.lstm_1_units_config,
            "dropout_1": 0.2,
            "lstm_2_units": self.lstm_2_units_config,
            "dropout_2": self.dropout_2_config,

            # ----------- optimization parameters
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class Seq2SeqRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search, only for LSTM.
    """

    def __init__(
            self,
            num_rand_samples=1,
            epochs=5,
            training_iteration=10,
            look_back=2,
            latent_dim=[32, 64, 128, 256],
            batch_size=[32, 64]):
        """
        __init__()
        Constructor.

        :param lstm_1_units: random search candidates for num of lstm_1_units
        :param lstm_2_units: grid search candidates for num of lstm_1_units
        :param batch_size: grid search candidates for batch size
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        # -- runtime params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- model params
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)
        self.latent_dim = hp.choice(latent_dim)
        self.dropout_config = hp.uniform(0.2, 0.5)

        # -- optimization params
        self.lr = hp.uniform(0.001, 0.01)
        self.batch_size = hp.grid_search(batch_size)
        self.epochs = epochs

    def search_space(self):
        return {
            "model": "Seq2Seq",
            "latent_dim": self.latent_dim,
            "dropout": self.dropout_config,

            # ----------- optimization parameters
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class MTNetGridRandomRecipe(Recipe):
    """
    Grid+Random Recipe for MTNet
    """

    def __init__(self,
                 num_rand_samples=1,
                 epochs=5,
                 training_iteration=10,
                 time_step=[3, 4],
                 long_num=[3, 4],
                 cnn_height=[2, 3],
                 cnn_hid_size=[32, 50, 100],
                 ar_size=[2, 3],
                 batch_size=[32, 64]):
        """
        __init__()
        Constructor.

        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        :param time_step: random search candidates for model param "time_step"
        :param long_num: random search candidates for model param "long_num"
        :param ar_size: random search candidates for model param "ar_size"
        :param batch_size: grid search candidates for batch size
        :param cnn_height: random search candidates for model param "cnn_height"
        :param cnn_hid_size: random search candidates for model param "cnn_hid_size"
        """
        super(self.__class__, self).__init__()
        # -- run time params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- optimization params
        self.lr = hp.uniform(0.001, 0.01)
        self.batch_size = hp.grid_search(batch_size)
        self.epochs = epochs

        # ---- model params
        self.cnn_dropout = hp.uniform(0.2, 0.5)
        self.rnn_dropout = hp.uniform(0.2, 0.5)
        self.time_step = hp.choice(time_step)
        self.long_num = hp.choice(long_num,)
        self.cnn_height = hp.choice(cnn_height)
        self.cnn_hid_size = hp.choice(cnn_hid_size)
        self.ar_size = hp.choice(ar_size)
        self.past_seq_len = hp.sample_from(
            lambda spec: (
                spec.config.long_num + 1) * spec.config.time_step)

    def search_space(self):
        return {
            "model": "MTNet",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "cnn_dropout": self.cnn_dropout,
            "rnn_dropout": self.rnn_dropout,
            "time_step": self.time_step,
            "long_num": self.long_num,
            "ar_size": self.ar_size,
            "past_seq_len": self.past_seq_len,
            "cnn_hid_size": self.cnn_hid_size,
            "cnn_height": self.cnn_height
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class TCNGridRandomRecipe(Recipe):
    """
    Grid+Random Recipe for TCN
    """
    # TODO: use some more generalized exp hyperparameters
    def __init__(self,
                 num_rand_samples=1,
                 training_iteration=40,
                 batch_size=[256, 512],
                 hidden_size=[32, 48],
                 levels=[6, 8],
                 kernel_size=[3, 5],
                 dropout=[0, 0.1],
                 lr=[0.001, 0.003]
                 ):
        """
        __init__()
        Constructor.

        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param batch_size: grid search candidates for batch size
        :param hidden_size: grid search candidates for hidden size of each layer
        :param levels: the number of layers
        :param kernel_size: the kernel size of each layer
        :param dropout: dropout rate (1 - keep probability)
        :param lr: learning rate
        """
        super(self.__class__, self).__init__()
        # -- run time params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- optimization params
        self.lr = hp.choice(lr)
        self.batch_size = hp.grid_search(batch_size)

        # ---- model params
        self.hidden_size = hp.grid_search(hidden_size)
        self.levels = hp.grid_search(levels)
        self.kernel_size = hp.grid_search(kernel_size)
        self.dropout = hp.choice(dropout)

    def search_space(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "nhid": self.hidden_size,
            "levels": self.levels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class RandomRecipe(Recipe):
    """
    Pure random sample Recipe. Often used as baseline.
    """

    def __init__(
            self,
            num_rand_samples=1,
            look_back=2,
            epochs=5,
            reward_metric=-0.05,
            training_iteration=10):
        """
        __init__()
        Constructor.

        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back:the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param reward_metric: the rewarding metric value, when reached, stop trial
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        self.num_samples = num_rand_samples
        self.reward_metric = reward_metric
        self.training_iteration = training_iteration
        self.epochs = epochs
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)

    def search_space(self):
        return {
            "model": hp.choice(["LSTM", "Seq2seq"]),
            # --------- Vanilla LSTM model parameters
            "lstm_1_units": hp.choice([8, 16, 32, 64, 128]),
            "dropout_1": hp.uniform(0.2, 0.5),
            "lstm_2_units": hp.choice([8, 16, 32, 64, 128]),
            "dropout_2": hp.uniform(0.2, 0.5),

            # ----------- Seq2Seq model parameters
            "latent_dim": hp.choice([32, 64, 128, 256]),
            "dropout": hp.uniform(0.2, 0.5),

            # ----------- optimization parameters
            "lr": hp.uniform(0.001, 0.01),
            "batch_size": hp.choice([32, 64, 1024]),
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


@deprecated('Please use `bigdl.orca.automl.hp` instead.')
class BayesRecipe(Recipe):
    """
    A Bayes search Recipe. (Experimental)
    """

    def __init__(
            self,
            num_samples=1,
            look_back=2,
            epochs=5,
            reward_metric=-0.05,
            training_iteration=5):
        """
        __init__()
        Constructor.

        :param num_samples: number of hyper-param configurations sampled
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param reward_metric: the rewarding metric value, when reached, stop trial
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        from bigdl.nano.utils.common import invalidInputError
        self.num_samples = num_samples
        self.reward_metric = reward_metric
        self.training_iteration = training_iteration
        self.epochs = epochs
        if isinstance(look_back, tuple) and len(look_back) == 2 and \
                isinstance(look_back[0], int) and isinstance(look_back[1], int):
            if look_back[1] < 2:
                invalidInputError(False,
                                  "The max look back value should be at least 2")
            if look_back[0] < 2:
                print("The input min look back value is smaller than 2. "
                      "We sample from range (2, {}) instead.".format(look_back[1]))
            self.bayes_past_seq_config = \
                {"past_seq_len_float": hp.uniform(look_back[0], look_back[1])}
        elif isinstance(look_back, int):
            if look_back < 2:
                invalidInputError(False,
                                  "look back value should not be smaller than 2. "
                                  "Current value is ", look_back)
            self.bayes_past_seq_config = {"past_seq_len": look_back}
        else:
            invalidInputError(False,
                              "look back is {}.\n "
                              "look_back should be either a tuple with 2 int values:"
                              " (min_len, max_len) or a single int".format(look_back))

    def search_space(self):
        total_params = {
            "epochs": self.epochs,
            "model": "LSTM",
            # --------- model parameters
            "lstm_1_units_float": hp.uniform(8, 128),
            "dropout_1": hp.uniform(0.2, 0.5),
            "lstm_2_units_float": hp.uniform(8, 128),
            "dropout_2": hp.uniform(0.2, 0.5),
            # ----------- optimization parameters
            "lr": hp.uniform(0.001, 0.1),
            "batch_size_float": hp.uniform(32, 128),
        }
        total_params.update(self.bayes_past_seq_config)
        return total_params


class XgbRegressorGridRandomRecipe(Recipe):
    """
    Grid + Random Recipe for XGBoost Regressor.
    """

    def __init__(
            self,
            num_rand_samples=1,
            n_estimators=[8, 15],
            max_depth=[10, 15],
            n_jobs=-1,
            tree_method='hist',
            random_state=2,
            seed=0,
            lr=(1e-4, 1e-1),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=[1, 2, 3],
            gamma=0,
            reg_alpha=0,
            reg_lambda=1):
        """
        Constructor. For XGBoost hyper parameters, refer to
        https://xgboost.readthedocs.io/en/latest/python/python_api.html for
        details.

        :param num_rand_samples: number of hyper-param configurations sampled
          randomly
        :param n_estimators: number of gradient boosted trees.
        :param max_depth: max tree depth
        :param n_jobs: number of parallel threads used to run xgboost.
        :param tree_method: specify which tree method to use.
        :param random_state: random number seed.
        :param seed: seed used to generate the folds
        :param lr: learning rate
        :param subsample: subsample ratio of the training instance
        :param colsample_bytree: subsample ratio of columns when constructing
          each tree.
        :param min_child_weight: minimum sum of instance weight(hessian)
          needed in a child.
        :param gamma: minimum loss reduction required to make a further
          partition on a leaf node of the tree.
        :param reg_alpha: L1 regularization term on weights (xgb’s alpha).
        :param reg_lambda: L2 regularization term on weights (xgb’s lambda).

        """
        super(self.__class__, self).__init__()

        self.num_samples = num_rand_samples
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.random_state = random_state
        self.seed = seed

        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.n_estimators = hp.grid_search(n_estimators)
        self.max_depth = hp.grid_search(max_depth)
        self.lr = hp.loguniform(lr[0], lr[-1])
        self.subsample = subsample
        self.min_child_weight = hp.choice(min_child_weight)

    def search_space(self):
        return {
            # -------- feature related parameters
            "model": "XGBRegressor",

            "imputation": hp.choice(["LastFillImpute", "FillZeroImpute"]),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "lr": self.lr
        }


class XgbRegressorSkOptRecipe(Recipe):
    """
    A recipe using SkOpt search algorithm for XGBoost Regressor.
    """

    def __init__(
            self,
            num_rand_samples=10,
            n_estimators_range=(50, 1000),
            max_depth_range=(2, 15),
            lr=(1e-4, 1e-1),
            min_child_weight=[1, 2, 3],
    ):
        """
        Constructor.

        :param num_rand_samples: number of hyper-param configurations sampled
          randomly
        :param n_estimators_range: range of number of gradient boosted trees.
        :param max_depth_range: range of max tree depth
        :param lr: learning rate
        :param min_child_weight: minimum sum of instance weight(hessian)
          needed in a child.
        """
        super(self.__class__, self).__init__()

        self.num_samples = num_rand_samples

        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.lr = hp.loguniform(lr[0], lr[1])
        self.min_child_weight = hp.choice(min_child_weight)

    def search_space(self):
        space = {
            "n_estimators": hp.randint(self.n_estimators_range[0],
                                       self.n_estimators_range[1]),
            "max_depth": hp.randint(self.max_depth_range[0],
                                    self.max_depth_range[1]),
            "min_child_weight": self.min_child_weight,
            "lr": self.lr
        }
        return space
