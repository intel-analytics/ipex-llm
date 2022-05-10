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

from enum import Enum


class SamplerType(Enum):
    """Types of Samplers. Sampler are used for generating hyper parameters."""

    TPE = 1  # the default
    CmaEs = 2
    Grid = 3
    Random = 4
    PartialFixed = 5
    NSGAII = 6  # multi-objective sampler
    MOTPE = 7  # multi-objective sampler


class PrunerType(Enum):
    """Types of Pruners. Pruners are used to stop non-promising trials early."""

    HyperBand = 1  # the default
    Median = 2
    Nop = 3
    Patient = 4
    Percentile = 5
    SuccessiveHalving = 6
    Threshold = 7


def create_hpo_backend(*args, **kwargs):
    """Create HPO Search Backend. Only Optuna is supported."""
    from bigdl.nano.deps.automl.hpo_api import create_optuna_backend
    return create_optuna_backend(*args, **kwargs)


def create_tfkeras_pruning_callback(*args, **kwargs):
    """Create Tensorflow Pruning Callback. Optuna Only."""
    from bigdl.nano.deps.automl.hpo_api import create_optuna_tfkeras_pruning_callback
    return create_optuna_tfkeras_pruning_callback(*args, **kwargs)


def create_pl_pruning_callback(*args, **kwargs):
    """Create PyTorchLightning Pruning Callback. Optuna Only."""
    from bigdl.nano.deps.automl.hpo_api import create_optuna_pl_pruning_callback
    return create_optuna_pl_pruning_callback(*args, **kwargs)
