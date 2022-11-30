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
import types
from abc import ABC, abstractmethod
from bigdl.dllib.utils.log4Error import invalidInputError

from typing import TYPE_CHECKING
from typing import Callable
if TYPE_CHECKING:
    from bigdl.dllib.keras.metrics import AUC as KerasAUC, MAE as KerasMAE,\
        MSE as KerasMSE, Accuracy as KerasAccuracy, Top5Accuracy as KerasTop5Accuracy,\
        BinaryAccuracy as KerasBinaryAccuracy, CategoricalAccuracy as KerasCategoricalAccuracy,\
        SparseCategoricalAccuracy as KerasSparseCategoricalAccuracy
    from bigdl.orca.learn.pytorch.pytorch_metrics import PytorchMetric, AUROC,\
        MAE as PyTorchMAE, MSE as PyTorchMSE, Accuracy as PyTorchAccuracy,\
        SparseCategoricalCrossEntropy as PyTorchSparseCategoricalCrossEntropy,\
        PrecisionRecallCurve as PyTorchPrecisionRecallCurve,\
        Recall as PyTorchRecall, Precision as PyTorchPrecision,\
        F1Score as PyTorchF1Score, ROC as PyTorchROC, Poisson as PyTorchPoisson,\
        KLDivergence as PyTorchKLDivergence, Top5Accuracy as PyTorchTop5Accuracy,\
        SparseCategoricalAccuracy as PyTorchSparseCategoricalAccuracy,\
        CategoricalCrossEntropy as PyTorchCategoricalCrossEntropy,\
        BinaryCrossEntropy as PyTorchBinaryCrossEntropy,\
        BinaryAccuracy as PyTorchBinaryAccuracy,\
        CategoricalAccuracy as PyTorchCategoricalAccuracy


class Metric(ABC):

    def get_metric(self, backend: str = "bigdl"):

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
            invalidInputError(False,
                              f"backend should be one of {valid_backends}, but got {backend}")
        return metric_impl

    def get_bigdl_metric(self):
        invalidInputError(False, "not implemented")

    def get_tf_metric(self):
        invalidInputError(False, "not implemented")

    def get_pytorch_metric(self):
        invalidInputError(False, "not implemented")

    def get_mxnet_metric(self):
        invalidInputError(False, "not implemented")

    @abstractmethod
    def get_name(self) -> str:
        pass

    @staticmethod
    def convert_metrics_list(metrics, backend: str = "bigdl"):
        if metrics is None:
            return None
        if not isinstance(metrics, list):
            metrics = [metrics]

        metric_impls = []
        for m in metrics:
            if isinstance(m, Metric):
                metric_impls.append(m.get_metric(backend))
            elif isinstance(m, types.FunctionType):
                customized_metric = CustomizedMetric(m)
                metric_impls.append(customized_metric.get_metric(backend))
            else:
                invalidInputError(False, "Only orca metrics and customized functions "
                                         "are supported, but get " + m.__class__.__name__)
        return metric_impls

    @staticmethod
    def convert_metrics_dict(metrics, backend: str = "bigdl"):
        if metrics is None:
            return {}
        if not isinstance(metrics, list):
            metrics = [metrics]

        metric_impls = {}

        for m in metrics:
            if isinstance(m, Metric):
                metric_impls[m.get_name()] = m.get_metric(backend)
            elif isinstance(m, types.FunctionType):
                my_metric = CustomizedMetric(m)
                metric_impls[my_metric.get_name()] = my_metric.get_metric(backend)
            else:
                invalidInputError(False, "Only orca metrics and customized functions "
                                         "are supported, but get " + m.__class__.__name__)
        return metric_impls


class CustomizedMetric(Metric):
    def __init__(self, compute_function: Callable) -> None:
        self.compute_function = compute_function

    def get_pytorch_metric(self) -> "PytorchMetric":
        from bigdl.orca.learn.pytorch.pytorch_metrics import PytorchMetric

        class Metric(PytorchMetric):
            def __init__(self, compute_function):
                self.batch_metric_value = 0
                self.compute_function = compute_function
                self.step = 0

            def __call__(self, preds, targets):
                self.batch_metric_value += self.compute_function(preds, targets)
                self.step += 1

            def compute(self):
                return self.batch_metric_value / self.step

        return Metric(self.compute_function)

    def get_name(self) -> str:
        return self.compute_function.__name__


class AUC(Metric):
    """
    Metric for binary(0/1) classification, support single label and multiple labels.

    # Arguments
    threshold_num: The number of thresholds. The quality of approximation
                   may vary depending on threshold_num.

    >>> meter = AUC(20)
    """

    def __init__(self, threshold_num: int = 200) -> None:
        self.threshold_num = threshold_num

    def get_bigdl_metric(self) -> "KerasAUC":
        from bigdl.dllib.keras.metrics import AUC as KerasAUC
        return KerasAUC(threshold_num=self.threshold_num)

    def get_pytorch_metric(self) -> "AUROC":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.AUROC()

    def get_name(self) -> str:
        return "AUC"


class MAE(Metric):
    """
    Metric for mean absoluate error, similar from MAE criterion

    >>> mae = MAE()

    """

    def get_bigdl_metric(self) -> "KerasMAE":
        from bigdl.dllib.keras.metrics import MAE as KerasMAE
        return KerasMAE()

    def get_pytorch_metric(self) -> "PyTorchMAE":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.MAE()

    def get_name(self) -> str:
        return "MAE"


class MSE(Metric):
    """
    Metric for mean square error, similar from MSE criterion

    >>> mse = MSE()

    """

    def get_bigdl_metric(self) -> "KerasMSE":
        from bigdl.dllib.keras.metrics import MSE as KerasMSE
        return KerasMSE()

    def get_pytorch_metric(self) -> "PyTorchMSE":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.MSE()

    def get_name(self) -> str:
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

    def __init__(self, zero_based_label: bool = True) -> None:
        self.zero_based_label = zero_based_label

    def get_bigdl_metric(self) -> "KerasAccuracy":
        from bigdl.dllib.keras.metrics import Accuracy as KerasAccuracy
        return KerasAccuracy(zero_based_label=self.zero_based_label)

    def get_pytorch_metric(self) -> "PyTorchAccuracy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        if not self.zero_based_label:
            invalidInputError(False,
                              "pytorch Accuracy does not support one based accuracy, "
                              "please set zero_based_label to True")
        return pytorch_metrics.Accuracy()

    def get_name(self) -> str:
        return "Accuracy"


class SparseCategoricalAccuracy(Metric):
    """
    Measures top1 accuracy for multi-class classification with sparse target.

    >>> acc = SparseCategoricalAccuracy()
    """

    def get_bigdl_metric(self) -> "KerasSparseCategoricalAccuracy":
        from bigdl.dllib.keras.metrics import \
            SparseCategoricalAccuracy as KerasSparseCategoricalAccuracy
        return KerasSparseCategoricalAccuracy()

    def get_pytorch_metric(self) -> "PyTorchSparseCategoricalAccuracy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.SparseCategoricalAccuracy()

    def get_name(self) -> str:
        return "SparseCategoricalAccuracy"


class CategoricalAccuracy(Metric):
    """
    Measures top1 accuracy for multi-class classification when target is one-hot encoded.

    >>> acc = CategoricalAccuracy()
    """

    def get_bigdl_metric(self) -> "KerasCategoricalAccuracy":
        from bigdl.dllib.keras.metrics import CategoricalAccuracy as KerasCategoricalAccuracy
        return KerasCategoricalAccuracy()

    def get_pytorch_metric(self) -> "PyTorchCategoricalAccuracy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.CategoricalAccuracy()

    def get_name(self) -> str:
        return "CategoricalAccuracy"


class BinaryAccuracy(Metric):
    """
    Measures top1 accuracy for binary classification with zero-based index.

    >>> acc = BinaryAccuracy()
    """

    def get_bigdl_metric(self) -> "KerasBinaryAccuracy":
        from bigdl.dllib.keras.metrics import BinaryAccuracy as KerasBinaryAccuracy
        return KerasBinaryAccuracy()

    def get_pytorch_metric(self) -> "PyTorchBinaryAccuracy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.BinaryAccuracy()

    def get_name(self) -> str:
        return "BinaryAccuracy"


class Top5Accuracy(Metric):
    """
    Measures top5 accuracy for multi-class classification.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.

    >>> acc = Top5Accuracy()
    """

    def get_bigdl_metric(self) -> "KerasTop5Accuracy":
        from bigdl.dllib.keras.metrics import Top5Accuracy as KerasTop5Accuracy
        return KerasTop5Accuracy()

    def get_pytorch_metric(self) -> "PyTorchTop5Accuracy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Top5Accuracy()

    def get_name(self) -> str:
        return "Top5Accuracy"


class BinaryCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are only 2 labels (0 and 1).

    >>> crossentropy = BinaryCrossEntropy()
    """

    def get_pytorch_metric(self) -> "PyTorchBinaryCrossEntropy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.BinaryCrossEntropy()

    def get_name(self) -> str:
        return "BinaryCrossEntropy"


class CategoricalCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are multiple labels (represented using one-hot vectors

    >>> crossentropy = CategoricalCrossEntropy()
    """

    def get_pytorch_metric(self) -> "PyTorchCategoricalCrossEntropy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.CategoricalCrossEntropy()

    def get_name(self) -> str:
        return "CategoricalCrossEntropy"


class SparseCategoricalCrossEntropy(Metric):
    """
    Calculates the cross entropy metric between the label and prediction when there
    are multiple labels (represented using integers)

    >>> crossentropy = SparseCategoricalCrossEntropy()
    """

    def get_pytorch_metric(self) -> "PyTorchSparseCategoricalCrossEntropy":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.SparseCategoricalCrossEntropy()

    def get_name(self) -> str:
        return "SparseCategoricalCrossEntropy"


class KLDivergence(Metric):
    """
    Calculates the Kullback-Liebler Divergence metric between the label and prediction.

    >>> div = KLDivergence()
    """

    def get_pytorch_metric(self) -> "PyTorchKLDivergence":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.KLDivergence()

    def get_name(self) -> str:
        return "KLDivergence"


class Poisson(Metric):
    """
    Calculates the Poisson metric between the label and prediction.

    >>> poisson = Poisson()
    """

    def get_pytorch_metric(self) -> "PyTorchPoisson":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Poisson()

    def get_name(self) -> str:
        return "Poisson"


class ROC(Metric):
    """
    Metric for binary(0/1) classification

    >>> meter = ROC()
    """

    def get_pytorch_metric(self) -> "PyTorchROC":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.ROC()

    def get_name(self) -> "str":
        return "ROC"


class F1Score(Metric):
    """
    Metric for binary(0/1) classification

    >>> meter = F1Score()
    """

    def get_pytorch_metric(self) -> "PyTorchF1Score":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.F1Score()

    def get_name(self) -> str:
        return "F1Score"


class Precision(Metric):
    """
    Metric for binary(0/1) classification

    >>> meter = Precision()
    """

    def get_pytorch_metric(self) -> "PyTorchPrecision":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Precision()

    def get_name(self) -> str:
        return "Precision"


class Recall(Metric):
    """
    Metric for binary(0/1) classification

    >>> meter = Recall()
    """

    def get_pytorch_metric(self) -> "PyTorchRecall":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.Recall()

    def get_name(self) -> str:
        return "Recall"


class PrecisionRecallCurve(Metric):
    """
    Metric for binary(0/1) classification

    >>> meter = PrecisionRecallCurve()
    """

    def get_pytorch_metric(self) -> "PyTorchPrecisionRecallCurve":
        from bigdl.orca.learn.pytorch import pytorch_metrics
        return pytorch_metrics.PrecisionRecallCurve()

    def get_name(self) -> str:
        return "PrecisionRecallCurve"
