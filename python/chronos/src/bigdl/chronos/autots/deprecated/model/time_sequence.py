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
from copy import deepcopy

from zoo.orca.automl.model.abstract import BaseModel, ModelBuilder
from zoo.chronos.model.VanillaLSTM import VanillaLSTM
from zoo.chronos.model.Seq2Seq import LSTMSeq2Seq
from zoo.chronos.model.MTNet_keras import MTNetKeras
from zoo.chronos.autots.deprecated.feature.utils import save_config
from zoo.chronos.autots.deprecated.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.chronos.autots.deprecated.preprocessing.impute import LastFillImpute, FillZeroImpute
from zoo.chronos.utils import deprecated
from zoo.orca.automl.metrics import Evaluator

import pandas as pd
import os
import tempfile
import shutil
import zipfile
import json

MODEL_MAP = {"LSTM": VanillaLSTM,
             "Seq2seq": LSTMSeq2Seq,
             "MTNet": MTNetKeras,
             }


class TSModelBuilder(ModelBuilder):
    def __init__(self,
                 dt_col,
                 target_cols,
                 future_seq_len=1,
                 extra_features_col=None,
                 drop_missing=True,
                 add_dt_features=True,
                 ):
        self.params = dict(
            dt_col=dt_col,
            target_cols=target_cols,
            future_seq_len=future_seq_len,
            extra_features_col=extra_features_col,
            drop_missing=drop_missing,
            add_dt_features=add_dt_features)

    def build(self, config):
        model = TimeSequenceModel.create(**self.params)
        model.setup(config)
        return model


class TimeSequenceModel(BaseModel):
    """
    Time Sequence Model integrates feature transformation model selection for time series
    forecasting.
    It has similar functionality with the TimeSequencePipeline.
    Note that to be compatible with load_ts_pipeline in TimeSequencePipeline,
    TimeSequenceModel should be able to restore with TimeSequenceModel().restore(checkpoint).
    TimeSequenceModel could be optimized if we deprecate load_ts_pipeline
     in future version.
    """
    def __init__(self,
                 feature_transformer=None):
        """
        Constructor of time sequence model
        """
        self.ft = feature_transformer if feature_transformer else TimeSequenceFeatureTransformer()
        self.model = None
        self.built = False
        self.config = None

    @classmethod
    def create(cls,
               dt_col,
               target_cols,
               future_seq_len=1,
               extra_features_col=None,
               drop_missing=True,
               add_dt_features=True):
        ft = TimeSequenceFeatureTransformer(
            future_seq_len=future_seq_len,
            dt_col=dt_col,
            target_col=target_cols,
            extra_features_col=extra_features_col,
            drop_missing=drop_missing,
            time_features=add_dt_features)
        return cls(feature_transformer=ft)

    def setup(self, config):
        # setup self.config, self.model, self.built
        self.config = config.copy()
        # add configs for model
        self.config["future_seq_len"] = self.ft.future_seq_len
        self.config["check_optional_config"] = False
        # for base keras model
        self.config["input_dim"] = self.ft.get_feature_dim()
        self.config["output_dim"] = self.ft.get_target_dim()
        # for base pytorch model
        self.config["input_feature_num"] = self.ft.get_feature_dim()
        self.config["output_feature_num"] = self.ft.get_target_dim()

        if not self.model:
            self.model = TimeSequenceModel._sel_model(self.config)
        # self.model.build(self.config)
        self.built = True

    def _process_data(self, data, mode="test"):
        df = deepcopy(data)
        imputer = None
        config = self.config.copy()
        if "imputation" in config:
            if config["imputation"] == "LastFillImpute":
                imputer = LastFillImpute()
            elif config["imputation"] == "FillZeroImpute":
                imputer = FillZeroImpute()
        if imputer:
            df = imputer.impute(df)

        if mode == "train":
            data_np = self.ft.fit_transform(df, **config)
        elif mode == "val":
            data_np = self.ft.transform(df, is_train=True)
        elif mode == "test":
            x, _ = self.ft.transform(df, is_train=False)
            data_np = x
        else:
            raise ValueError(f"Mode should be among ['train', 'val', 'test']. Got {mode}")
        return data_np

    def fit_eval(self, data, validation_data=None, **kwargs):
        """
        fit for one iteration
        :param data: pandas DataFrame
        :param validation_data: pandas DataFrame, data used for validation.
        If this is specified, validation result will be the optimization target for automl.
        Otherwise, train metric will be the optimization target.
        :return: the resulting metric
        """
        assert self.built, "You must call setup or restore before calling fit_eval"
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"We only support data of pd.DataFrame. "
                             f"Got data of {data.__class__.__name__}")
        if validation_data is not None and not isinstance(validation_data, pd.DataFrame):
            raise ValueError(f"We only support validation_data of pd.DataFrame. "
                             f"Got validation_data of {data.__class__.__name__}")
        data_np = self._process_data(data, mode="train")
        is_val_valid = isinstance(validation_data, pd.DataFrame) and not validation_data.empty
        if is_val_valid:
            val_data_np = self._process_data(data, mode="val")
        else:
            val_data_np = None

        return self.model.fit_eval(data=data_np,
                                   validation_data=val_data_np,
                                   **kwargs)

    def fit_incr(self, data, validation_data=None, **kwargs):
        assert self.built, "You must call setup or restore before calling fit_eval"
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"We only support data of pd.DataFrame. "
                             f"Got data of {data.__class__.__name__}")
        if validation_data is not None and not isinstance(validation_data, pd.DataFrame):
            raise ValueError(f"We only support validation_data of pd.DataFrame. "
                             f"Got validation_data of {data.__class__.__name__}")
        data_np = self._process_data(data, mode="val")
        is_val_valid = isinstance(validation_data, pd.DataFrame) and not validation_data.empty
        if is_val_valid:
            val_data_np = self._process_data(data, mode="val")
        else:
            # this is a work around since pytorch base model must include validation data for
            # fit_eval. We may need to optimize automl base model interface.
            val_data_np = data_np

        return self.model.fit_eval(data=data_np,
                                   validation_data=val_data_np,
                                   **kwargs)

    @staticmethod
    def _sel_model(config, verbose=0):
        model_name = config.get("model", "LSTM")
        model = MODEL_MAP[model_name](
            check_optional_config=config.get("check_optional_config", False))
        if verbose != 0:
            print(model_name, "is selected.")
        return model

    def evaluate(self, df, metric=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        if isinstance(metric, str):
            metric = [metric]
        x, y = self._process_data(df, mode="val")
        y_pred = self.model.predict(x)
        y_unscale, y_pred_unscale = self.ft.post_processing(df, y_pred, is_train=True)
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            multioutput = 'uniform_average'
        else:
            multioutput = 'raw_values'
        return [Evaluator.evaluate(m, y_unscale, y_pred_unscale, multioutput=multioutput)
                for m in metric]

    def predict(self, df):
        """
        Prediction on x.
        :param df: input
        :return: predicted y
        """
        data_np = self._process_data(df, mode="test")
        y_pred = self.model.predict(data_np)
        output = self.ft.post_processing(df, y_pred, is_train=False)
        return output

    def predict_with_uncertainty(self, df, n_iter=100):
        data_np = self._process_data(df, mode="test")
        y_pred, y_pred_uncertainty = self.model.predict_with_uncertainty(x=data_np, n_iter=n_iter)
        output = self.ft.post_processing(df, y_pred, is_train=False)
        uncertainty = self.ft.unscale_uncertainty(y_pred_uncertainty)
        return output, uncertainty

    def save(self, checkpoint_file):
        file_dirname = os.path.dirname(os.path.abspath(checkpoint_file))
        if file_dirname and not os.path.exists(file_dirname):
            os.makedirs(file_dirname)

        dirname = tempfile.mkdtemp(prefix="automl_save_")
        try:
            ppl_model = os.path.join(dirname, "model.ckpt")
            ppl_config = os.path.join(dirname, "config.json")
            self.ft.save(ppl_config, replace=True)
            self.model.save(ppl_model)
            save_config(ppl_config, self.config)

            with zipfile.ZipFile(checkpoint_file, 'w') as f:
                for dirpath, dirnames, filenames in os.walk(dirname):
                    for filename in filenames:
                        f.write(os.path.join(dirpath, filename), filename)
            assert os.path.isfile(checkpoint_file)
        finally:
            shutil.rmtree(dirname)

    def restore(self, checkpoint_file):
        dirname = tempfile.mkdtemp(prefix="automl_save_")
        try:
            with zipfile.ZipFile(checkpoint_file) as zf:
                zf.extractall(dirname)
            ppl_model = os.path.join(dirname, "model.ckpt")
            ppl_config = os.path.join(dirname, "config.json")

            with open(ppl_config, "r") as input_file:
                self.config = json.load(input_file)
            self.model = TimeSequenceModel._sel_model(self.config)
            self.model.restore(ppl_model)
            self.ft.restore(**self.config)
            self.built = True
        finally:
            shutil.rmtree(dirname)

    def _get_required_parameters(self):
        return self.model._get_required_parameters()

    def _get_optional_parameters(self):
        return self.model._get_optional_parameters()
