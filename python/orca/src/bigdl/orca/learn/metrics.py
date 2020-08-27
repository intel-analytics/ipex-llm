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
from abc import ABC, abstractmethod


class Metrics(ABC):
    @abstractmethod
    def get_metrics(self):
        pass

    @staticmethod
    def convert_metrics_list(metrics):
        if metrics is None:
            return None
        if isinstance(metrics, list):
            keras_metrics = []
            for m in metrics:
                if isinstance(m, Metrics):
                    keras_metrics.append(m.get_metrics())
                else:
                    keras_metrics.append(m)
            return keras_metrics
        else:
            if isinstance(metrics, Metrics):
                return metrics.get_metrics()
            else:
                raise ValueError("Only orca metrics are supported, but get " +
                                 metrics.__class__.__name__)


class AUC(Metrics):
    """
    Metric for binary(0/1) classification, support single label and multiple labels.

    # Arguments
    threshold_num: The number of thresholds. The quality of approximation
                   may vary depending on threshold_num.

    >>> meter = AUC(20)
    """
    def __init__(self, threshold_num=200):
        from zoo.pipeline.api.keras.metrics import AUC as KerasAUC
        self.metrics = KerasAUC(threshold_num=threshold_num)

    def get_metrics(self):
        return self.metrics


class MAE(Metrics):
    """
    Metric for mean absoluate error, similar from MAE criterion

    >>> mae = MAE()

    """
    def __init__(self):
        from zoo.pipeline.api.keras.metrics import MAE as KerasMAE
        self.metrics = KerasMAE()

    def get_metrics(self):
        return self.metrics


class Accuracy(Metrics):
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
        from zoo.pipeline.api.keras.metrics import Accuracy as KerasAccuracy
        self.metrics = KerasAccuracy(zero_based_label=zero_based_label)

    def get_metrics(self):
        return self.metrics


class SparseCategoricalAccuracy(Metrics):
    """
    Measures top1 accuracy for multi-class classification with sparse target.

    >>> acc = SparseCategoricalAccuracy()
    """
    def __init__(self):
        from zoo.pipeline.api.keras.metrics import \
            SparseCategoricalAccuracy as KerasSparseCategoricalAccuracy
        self.metrics = KerasSparseCategoricalAccuracy()

    def get_metrics(self):
        return self.metrics


class CategoricalAccuracy(Metrics):
    """
    Measures top1 accuracy for multi-class classification when target is one-hot encoded.

    >>> acc = CategoricalAccuracy()
    """
    def __init__(self):
        from zoo.pipeline.api.keras.metrics import CategoricalAccuracy as KerasCategoricalAccuracy
        self.metrics = KerasCategoricalAccuracy()

    def get_metrics(self):
        return self.metrics


class BinaryAccuracy(Metrics):
    """
    Measures top1 accuracy for binary classification with zero-based index.

    >>> acc = BinaryAccuracy()
    """
    def __init__(self):
        from zoo.pipeline.api.keras.metrics import BinaryAccuracy as KerasBinaryAccuracy
        self.metrics = KerasBinaryAccuracy()

    def get_metrics(self):
        return self.metrics


class Top5Accuracy(Metrics):
    """
    Measures top5 accuracy for multi-class classification.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.

    >>> acc = Top5Accuracy()
    """
    def __init__(self):
        from zoo.pipeline.api.keras.metrics import Top5Accuracy as KerasTop5Accuracy
        self.metrics = KerasTop5Accuracy()

    def get_metrics(self):
        return self.metrics
