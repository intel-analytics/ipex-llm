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
from abc import ABC, abstractmethod


class Metric(ABC):

    def get_metric(self, backend="bigdl"):

        if backend == "bigdl":
            metric_impl = self.get_bigdl_metric()
        elif backend == "pytorch":
            metric_impl = self.get_pytorch_metric()
        elif backend == "tf":
            metric_impl = self.get_tf_metric()
        elif backend == "mxnet":
            metric_impl = self.get_mxnet_metric()
        else:
            valid_backends = {
                "bigdl",
                "pytorch",
                "tf",
                "mxnet"
            }
            raise ValueError(f"backend should be one of {valid_backends}, but got {backend}")
        return metric_impl

    def get_bigdl_metric(self):
        raise NotImplementedError()

    def get_tf_metric(self):
        raise NotImplementedError()

    def get_pytorch_metric(self):
        raise NotImplementedError()

    def get_mxnet_metric(self):
        raise NotImplementedError()

    @abstractmethod
    def get_name(self):
        pass

    @staticmethod
    def convert_metrics_list(metrics, backend="bigdl"):
        if metrics is None:
            return None
        if isinstance(metrics, list):
            metric_impls = []
            for m in metrics:
                if isinstance(m, Metric):
                    metric_impls.append(m.get_metric(backend))
                else:
                    raise ValueError("Only orca metrics are supported, but get " +
                                     m.__class__.__name__)
            return metric_impls
        else:
            if isinstance(metrics, Metric):
                return [metrics.get_metric(backend)]

    @staticmethod
    def convert_metrics_dict(metrics, backend="bigdl"):
        if metrics is None:
            return {}
        if isinstance(metrics, list):
            metric_impls = {}
            for m in metrics:
                metric_impls[m.get_name()] = m.get_metric(backend)
            return metric_impls
        else:
            if isinstance(metrics, Metric):
                return {metrics.get_name(): metrics.get_metric(backend)}
            else:
                raise ValueError("Only orca metrics are supported, but get " +
                                 metrics.__class__.__name__)


class AUC(Metric):
    """
    Metric for binary(0/1) classification, support single label and multiple labels.

    # Arguments
    threshold_num: The number of thresholds. The quality of approximation
                   may vary depending on threshold_num.

    >>> meter = AUC(20)
    """
    def __init__(self, threshold_num=200):
        self.threshold_num = threshold_num

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import AUC as KerasAUC
        return KerasAUC(threshold_num=self.threshold_num)

    def get_name(self):
        return "AUC"


class MAE(Metric):
    """
    Metric for mean absoluate error, similar from MAE criterion

    >>> mae = MAE()

    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import MAE as KerasMAE
        return KerasMAE()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.MAE()

    def get_name(self):
        return "MAE"


class MSE(Metric):
    """
    Metric for mean square error, similar from MSE criterion

    >>> mse = MSE()

    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import MSE as KerasMSE
        return KerasMSE()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.MSE()

    def get_name(self):
        return "MSE"


class Accuracy(Metric):
    """
    Measures top1 accuracy for multi-class classification
    or accuracy for binary classification.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.
                      Note that this only takes effect for multi-class classification.
                      For binary classification, labels ought to be 0 or 1.

    >>> acc = Accuracy()
    """
    def __init__(self, zero_based_label=True):
        self.zero_based_label = zero_based_label

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import Accuracy as KerasAccuracy
        return KerasAccuracy(zero_based_label=self.zero_based_label)

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        if not self.zero_based_label:
            raise ValueError("pytorch Accuracy does not support one based accuracy, "
                             "please set zero_based_label to True")
        return pytorch_metrics.Accuracy()

    def get_name(self):
        return "Accuracy"


class SparseCategoricalAccuracy(Metric):
    """
    Measures top1 accuracy for multi-class classification with sparse target.

    >>> acc = SparseCategoricalAccuracy()
    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import \
            SparseCategoricalAccuracy as KerasSparseCategoricalAccuracy
        return KerasSparseCategoricalAccuracy()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.SparseCategoricalAccuracy()

    def get_name(self):
        return "SparseCategoricalAccuracy"


class CategoricalAccuracy(Metric):
    """
    Measures top1 accuracy for multi-class classification when target is one-hot encoded.

    >>> acc = CategoricalAccuracy()
    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import CategoricalAccuracy as KerasCategoricalAccuracy
        return KerasCategoricalAccuracy()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.CategoricalAccuracy()

    def get_name(self):
        return "CategoricalAccuracy"


class BinaryAccuracy(Metric):
    """
    Measures top1 accuracy for binary classification with zero-based index.

    >>> acc = BinaryAccuracy()
    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import BinaryAccuracy as KerasBinaryAccuracy
        return KerasBinaryAccuracy()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.BinaryAccuracy()

    def get_name(self):
        return "BinaryAccuracy"


class Top5Accuracy(Metric):
    """
    Measures top5 accuracy for multi-class classification.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.

    >>> acc = Top5Accuracy()
    """

    def get_bigdl_metric(self):
        from bigdl.dllib.keras.metrics import Top5Accuracy as KerasTop5Accuracy
        return KerasTop5Accuracy()

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Top5Accuracy()

    def get_name(self):
        return "Top5Accuracy"


class BinaryCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are only 2 labels (0 and 1).

    >>> crossentropy = BinaryCrossEntropy()
    """

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.BinaryCrossEntropy()

    def get_name(self):
        return "BinaryCrossEntropy"


class CategoricalCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are multiple labels (represented using one-hot vectors

    >>> crossentropy = CategoricalCrossEntropy()
    """

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.CategoricalCrossEntropy()

    def get_name(self):
        return "CategoricalCrossEntropy"


class SparseCategoricalCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are multiple labels (represented using integers)

    >>> crossentropy = SparseCategoricalCrossEntropy()
    """

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.SparseCategoricalCrossEntropy()

    def get_name(self):
        return "SparseCategoricalCrossEntropy"


class KLDivergence(Metric):
    """
    Calculates the Kullback-Liebler Divergence metric between the label and prediction.

    >>> div = KLDivergence()
    """

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.KLDivergence()

    def get_name(self):
        return "KLDivergence"


class Poisson(Metric):
    """
    Calculates the Poisson metric between the label and prediction.

    >>> poisson = Poisson()
    """

    def get_pytorch_metric(self):
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Poisson()

    def get_name(self):
        return "Poisson"
