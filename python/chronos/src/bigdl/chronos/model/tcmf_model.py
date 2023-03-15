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

from bigdl.chronos.model.tcmf import DeepGLO
from bigdl.orca.automl.metrics import Evaluator
from bigdl.orca.automl.model.abstract import BaseModel
from bigdl.orca.data import SparkXShards, XShards
import pickle
import numpy as np
import pandas as pd


class TCMF(BaseModel):
    """
    MF regularized TCN + TCN. This version is not for automated searching yet.
    """

    def __init__(self):
        """
        Initialize hyper parameters
        :param check_optional_config:
        :param future_seq_len:
        """
        # models
        self.model = None
        self.model_init = False

    def build(self, config):
        """
        build the models and initialize.
        :param config: hyper parameters for building the model
        :return:
        """
        self.model = DeepGLO(
            vbsize=config.get("vbsize", 128),
            hbsize=config.get("hbsize", 256),
            num_channels_X=config.get("num_channels_X", [32, 32, 32, 32, 32, 1]),
            num_channels_Y=config.get("num_channels_Y", [16, 16, 16, 16, 16, 1]),
            kernel_size=config.get("kernel_size", 7),
            dropout=config.get("dropout", 0.1),
            rank=config.get("rank", 64),
            kernel_size_Y=config.get("kernel_size_Y", 7),
            lr=config.get("learning_rate", 0.0005),
            normalize=config.get("normalize", False),
            use_time=config.get("use_time", True),
            svd=config.get("svd", True),
            forward_cov=False
        )
        self.model_init = True

    def fit_eval(self, data, verbose=0, num_workers=None, **config):
        """
        Fit on the training data from scratch.
        Since the rolling process is very customized in this model,
        we enclose the rolling process inside this method.
        :param data: could be a tuple with numpy ndarray with form (x, y)
               x: training data, an array in shape (nd, Td),
                  nd is the number of series, Td is the time dimension
               y: None. target is extracted from x directly
        :param verbose:
        :param num_workers: number of workers to use.
        :return: the evaluation metric value
        """
        x = data[0]
        if not self.model_init:
            self.build(config)
        if num_workers is None:
            num_workers = TCMF.get_default_num_workers()
        covariates = config.get('covariates', None)
        dti = config.get("dti", None)
        self._check_covariates_dti(covariates=covariates, dti=dti, ts_len=x.shape[1])
        val_loss = self.model.train_all_models(x,
                                               val_len=config.get("val_len", 24),
                                               start_date=config.get("start_date", "2020-4-1"),
                                               freq=config.get("freq", "1H"),
                                               covariates=covariates,
                                               dti=dti,
                                               period=config.get("period", 24),
                                               init_epochs=config.get("init_FX_epoch", 100),
                                               alt_iters=config.get("alt_iters", 10),
                                               y_iters=config.get("y_iters", 10),
                                               max_FX_epoch=config.get("max_FX_epoch", 300),
                                               max_TCN_epoch=config.get("max_TCN_epoch", 300),
                                               num_workers=num_workers,
                                               )
        return {"val_loss": val_loss}

    def fit_incremental(self, x, covariates_new=None, dti_new=None):
        """
        Incremental fitting given a pre-trained model.
        :param x: incremental data
        :param covariates_new: covariates corresponding to the incremental x
        :param dti_new: dti corresponding to the incremental x
        :return:
        """
        from bigdl.nano.utils.common import invalidInputError
        if x is None:
            invalidInputError(False,
                              "Input invalid x of None")
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling "
                              "fit_incremental")
        self._check_covariates_dti(covariates=covariates_new, dti=dti_new, ts_len=x.shape[1],
                                   method_name='fit_incremental')
        self.model.inject_new(x,
                              covariates_new=covariates_new,
                              dti_new=dti_new)

    @staticmethod
    def get_default_num_workers():
        from bigdl.orca.ray import OrcaRayContext
        try:
            ray_ctx = OrcaRayContext.get(initialize=False)
            num_workers = ray_ctx.num_ray_nodes
        except:
            num_workers = 1
        return num_workers

    def predict(self, x=None, horizon=24, mc=False,
                future_covariates=None,
                future_dti=None,
                num_workers=None):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :param mc:
        :param future_covariates: covariates corresponding to future horizon steps data to predict.
        :param future_dti: dti corresponding to future horizon steps data to predict.
        :param num_workers: the number of workers to use. Note that there has to be an activate
               OrcaRayContext if num_workers > 1.
        :return:
        """
        from bigdl.nano.utils.common import invalidInputError
        if x is not None:
            invalidInputError(False,
                              "We don't support input x directly.")
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling predict")
        self._check_covariates_dti(covariates=future_covariates, dti=future_dti, ts_len=horizon,
                                   method_name="predict")
        if num_workers is None:
            num_workers = TCMF.get_default_num_workers()
        if num_workers > 1:
            import ray
            from bigdl.orca.ray import OrcaRayContext
            try:
                OrcaRayContext.get(initialize=False)
            except:
                try:
                    # detect whether ray has been started.
                    ray.put(None)
                except:
                    invalidInputError(False,
                                      f"There must be an activate ray context while running with "
                                      f"{num_workers} workers. You can either start and init a "
                                      f"RayContext by init_orca_context(..., init_ray_on_spark="
                                      f"True) or start Ray with ray.init()")

        out = self.model.predict_horizon(
            future=horizon,
            bsize=90,
            num_workers=num_workers,
            future_covariates=future_covariates,
            future_dti=future_dti,
        )
        return out[:, -horizon::]

    def evaluate(self, x=None, y=None, metrics=None, target_covariates=None,
                 target_dti=None, num_workers=None):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: We don't support input x currently.
        :param y: target. We interpret the second dimension of y as the horizon length for
            evaluation.
        :param metrics: a list of metrics in string format
        :param target_covariates: covariates corresponding to target_value.
            2-D ndarray or None.
            The shape of ndarray should be (r, horizon), where r is the number of covariates.
            Global covariates for all time series. If None, only default time coveriates will be
            used while use_time is True. If not, the time coveriates used is the stack of input
            covariates and default time coveriates.
        :param target_dti: dti corresponding to target_value.
            DatetimeIndex or None.
            If None, use default fixed frequency DatetimeIndex generated with the last date of x in
            fit and freq.
        :param num_workers: the number of workers to use in evaluate. It defaults to 1.
        :return: a list of metric evaluation results
        """
        from bigdl.nano.utils.common import invalidInputError
        if x is not None:
            invalidInputError(False,
                              "We don't support input x directly.")
        if y is None:
            invalidInputError(False,
                              "Input invalid y of None")
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling predict")
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
            horizon = 1
        else:
            horizon = y.shape[1]
        result = self.predict(x=None, horizon=horizon,
                              future_covariates=target_covariates,
                              future_dti=target_dti,
                              num_workers=num_workers)

        if y.shape[1] == 1:
            multioutput = 'uniform_average'
        else:
            multioutput = 'raw_values'
        return [Evaluator.evaluate(m, y, result, multioutput=multioutput) for m in metrics]

    def save(self, model_file):
        pickle.dump(self.model, open(model_file, "wb"))

    def restore(self, model_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.model_init = True

    def _get_optional_parameters(self):
        return {}

    def _get_required_parameters(self):
        return {}

    def _check_covariates_dti(self, covariates=None, dti=None, ts_len=24, method_name='fit'):
        from bigdl.nano.utils.common import invalidInputError
        if covariates is not None and not isinstance(covariates, np.ndarray):
            invalidInputError(False,
                              f"Input covariates must be a ndarray. Got ${type(covariates)}")
        if covariates is not None and not covariates.ndim == 2:
            invalidInputError(False,
                              f"You should input a 2-D ndarray of covariates. But Got dimension"
                              f" of ${covariates.ndim}")
        if covariates is not None and not covariates.shape[1] == ts_len:
            invalidInputError(False,
                              f"The second dimension shape of covariates should be {ts_len}, "
                              f"but got {covariates.shape[1]} instead.")
        if dti is not None and not isinstance(dti, pd.DatetimeIndex):
            invalidInputError(False,
                              f"Input dti must be a pandas DatetimeIndex. Got ${type(dti)}")
        if dti is not None and len(dti) != ts_len:
            invalidInputError(False,
                              f"Input dti length should be equal to {ts_len}, "
                              f"but got {len(dti)} instead.")

        if method_name != 'fit':
            # covariates and dti should be consistent with that in fit
            if self.model.covariates is None and covariates is not None:
                invalidInputError(False,
                                  f"Find valid covariates in {method_name} but invalid covariates "
                                  f"in fit. Please keep them in consistence!")
            if self.model.covariates is not None and covariates is None:
                invalidInputError(False,
                                  f"Find valid covariates in fit but invalid covariates in "
                                  f"{method_name}. Please keep them in consistence!")
            if self.model.covariates is not None \
                    and self.model.covariates.shape[0] != covariates.shape[0]:
                invalidInputError(False,
                                  f"The input covariates number in {method_name} should be"
                                  f" the same as the input covariates number in fit. Got"
                                  f" {covariates.shape[0]}"
                                  f"and {self.model.covariates.shape[0]} respectively.")
            if self.model.dti is None and dti is not None:
                invalidInputError(False,
                                  f"Find valid dti in {method_name} but invalid dti in fit. "
                                  f"Please keep them in consistence!")
            if self.model.dti is not None and dti is None:
                invalidInputError(False,
                                  f"Find valid dti in fit but invalid dti in {method_name}. "
                                  f"Please keep them in consistence!")


class ModelWrapper(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def is_xshards_distributed(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass


class TCMFXshardsModelWrapper(ModelWrapper):

    def __init__(self, config):
        self.internal = None
        self.config = config

    def fit(self, x, num_workers=None, **fit_params):
        from bigdl.nano.utils.common import invalidInputError
        if num_workers:
            invalidInputError(False,
                              "We don't support passing num_workers in fit "
                              "with input of xShards of builtins.dict")

        def orca_train_model(d, config):
            tcmf = TCMF()
            tcmf.build(config)
            id_arr, train_data = split_id_and_data(d, True)
            tcmf.fit_eval((train_data, None), **fit_params)
            return [id_arr, tcmf]

        if isinstance(x, SparkXShards):
            if x._get_class_name() == "builtins.dict":
                self.internal = x.transform_shard(orca_train_model, self.config)
            else:
                invalidInputError(False,
                                  "value of x should be an xShards of builtins.dict, "
                                  "but is an xShards of " + x._get_class_name())
        else:
            invalidInputError(False,
                              "value of x should be an xShards of builtins.dict, "
                              "but isn't an xShards")

    def fit_incremental(self, x_incr, covariates_incr=None, dti_incr=None):
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False, "fit_incremental not implemented")

    def evaluate(self, y, metric=None, target_covariates=None,
                 target_dti=None, num_workers=None):
        """
        Evaluate the model
        :param x: input
        :param y: target
        :param metric:
        :param num_workers:
        :param target_covariates:
        :param target_dti:
        :return: a list of metric evaluation results
        """
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False, "not implemented")

    def predict(self, horizon=24,
                future_covariates=None,
                future_dti=None,
                num_workers=None):
        """
        Prediction.
        :param horizon:
        :param future_covariates: covariates corresponding to future horizon steps data to predict.
        :param future_dti: dti corresponding to future horizon steps data to predict.
        :param num_workers
        :return: result
        """
        from bigdl.nano.utils.common import invalidInputError
        if num_workers and num_workers != 1:
            invalidInputError(False,
                              "We don't support passing num_workers in predict "
                              "with input of xShards of dict")

        def orca_predict(data):
            id_arr = data[0]
            tcmf = data[1]
            predict_results = tcmf.predict(x=None, horizon=horizon,
                                           future_covariates=future_covariates,
                                           future_dti=future_dti,)
            result = dict()
            result['id'] = id_arr
            result["prediction"] = predict_results
            return result

        return self.internal.transform_shard(orca_predict)

    def is_xshards_distributed(self):
        return True

    def save(self, model_path):
        """
        save model to file.
        :param model_path: the model file path to be saved to.
        :return:
        """
        if self.internal is not None:
            self.internal.save_pickle(model_path)

    def load(self, model_path, minPartitions=None):
        """
        restore model from model file and config.
        :param model_path: the model file
        :return: the restored model
        """
        self.internal = XShards.load_pickle(model_path, minPartitions=minPartitions)


class TCMFNdarrayModelWrapper(ModelWrapper):

    def __init__(self, config):
        self.internal = TCMF()
        self.config = config
        self.internal.build(self.config)
        self.id_arr = None

    def fit(self, x, num_workers=None, **fit_params):
        from bigdl.nano.utils.common import invalidInputError
        if isinstance(x, dict):
            self.id_arr, train_data = split_id_and_data(x, False)
            self.internal.fit_eval((train_data, None), num_workers=num_workers, **fit_params)
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "value of x should be a dict of ndarray")

    def _rearrange_data_by_id(self, id_new, data_new, method_name="fit_incremental"):
        from bigdl.nano.utils.common import invalidInputError
        if np.array_equal(self.id_arr, id_new) or id_new is None:
            return data_new
        if self.id_arr is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              f"Got valid id in {method_name} and invalid id in fit.")
        if set(id_new) != set(self.id_arr):
            invalidInputError(False,
                              f"The input ids in {method_name} differs from input ids in fit.")
        return data_new[[id_new.index(_) for _ in self.id_arr]]

    def fit_incremental(self, x_incr, covariates_incr=None, dti_incr=None):
        """
        incrementally fit the model. Note that we only incrementally fit X_seq (TCN in global model)
        :param x_incr: 2-D numpy array in shape (n, T_incr), where n is the number of target time
        series, T_incr is the number of time steps incremented.
            incremental data to be fitted.
        :param covariates_incr: covariates corresponding to x_incr. 2-D ndarray or None.
            The shape of ndarray should be (r, T_incr), where r is the number of covariates.
            Global covariates for all time series. If None, only default time coveriates will be
            used while use_time is True. If not, the time coveriates used is the stack of input
            covariates and default time coveriates.
        :param dti_incr: dti corresponding to the x_incr. DatetimeIndex or None.
            If None, use default fixed frequency DatetimeIndex generated with the last date of x in
            fit and freq.
        :return:
        """
        if isinstance(x_incr, dict):
            incr_id_arr, incr_train_data = split_id_and_data(x_incr, False)
            incr_train_data = self._rearrange_data_by_id(id_new=incr_id_arr,
                                                         data_new=incr_train_data,
                                                         method_name="fit_incremental")
            self.internal.fit_incremental(incr_train_data,
                                          covariates_new=covariates_incr,
                                          dti_new=dti_incr)
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "value of x should be a dict of ndarray")

    def evaluate(self, y, metric=None, target_covariates=None,
                 target_dti=None, num_workers=None):
        """
        Evaluate the model
        :param y: target
        :param metric:
        :param target_covariates:
        :param target_dti
        :param num_workers:
        :return: a list of metric evaluation results
        """
        if isinstance(y, dict):
            id_arr, y = split_id_and_data(y, False)
            y = self._rearrange_data_by_id(id_new=id_arr, data_new=y, method_name='evaluate')
            return self.internal.evaluate(y=y, metrics=metric,
                                          target_covariates=target_covariates,
                                          target_dti=target_dti,
                                          num_workers=num_workers)
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "value of y should be a dict of ndarray")

    def predict(self, horizon=24,
                future_covariates=None,
                future_dti=None,
                num_workers=None):
        """
        Prediction.
        :param horizon
        :param future_covariates: covariates corresponding to future horizon steps data to predict.
        :param future_dti: dti corresponding to future horizon steps data to predict.
        :param num_workers
        :return: result
        """
        pred = self.internal.predict(horizon=horizon, num_workers=num_workers,
                                     future_covariates=future_covariates,
                                     future_dti=future_dti,)
        result = dict()
        if self.id_arr is not None:
            result['id'] = self.id_arr
        result["prediction"] = pred
        return result

    def is_xshards_distributed(self):
        return False

    def save(self, model_path):
        """
        save model to file.
        :param model_path: the model file path to be saved to.
        :return:
        """
        with open(model_path + '/id.pkl', 'wb') as f:
            pickle.dump(self.id_arr, f)
        self.internal.save(model_path + "/model")

    def load(self, model_path):
        """
        restore model from model file and config.
        :param model_path: the model file
        :return: the restored model
        """
        self.internal = TCMF()
        with open(model_path + '/id.pkl', 'rb') as f:
            self.id_arr = pickle.load(f)
        self.internal.restore(model_path + "/model")


def split_id_and_data(d, is_xshards_distributed=False):
    from bigdl.nano.utils.common import invalidInputError
    if 'y' in d:
        train_data = d['y']
        if not isinstance(train_data, np.ndarray):
            invalidInputError(False,
                              "the value of y should be an ndarray")
    else:
        invalidInputError(False,
                          "key `y` doesn't exist in x")
    id_arr = None
    if 'id' in d:
        id_arr = d['id']
        if len(id_arr) != train_data.shape[0]:
            invalidInputError(False,
                              "the length of the id array should be equal to the number of "
                              "rows in the y")
    elif is_xshards_distributed:
        invalidInputError(False,
                          "key `id` doesn't exist in x")
    return id_arr, train_data
