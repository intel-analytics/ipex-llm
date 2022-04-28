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
import optuna


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


class OptunaBackend(object):
    """A Wrapper to shield user from Optuna specific configurations and API\
      Later may support other HPO search engines."""

    pruner_map = {
        PrunerType.HyperBand: optuna.pruners.HyperbandPruner,
        PrunerType.Median: optuna.pruners.MedianPruner,
        PrunerType.Nop: optuna.pruners.NopPruner,
        PrunerType.Patient: optuna.pruners.PatientPruner,
        PrunerType.Percentile: optuna.pruners.PercentilePruner,
        PrunerType.SuccessiveHalving: optuna.pruners.SuccessiveHalvingPruner,
        PrunerType.Threshold: optuna.pruners.ThresholdPruner,
    }

    sampler_map = {
        SamplerType.TPE: optuna.samplers.TPESampler,
        SamplerType.CmaEs: optuna.samplers.CmaEsSampler,
        SamplerType.Grid: optuna.samplers.GridSampler,
        SamplerType.Random: optuna.samplers.RandomSampler,
        SamplerType.PartialFixed: optuna.samplers.PartialFixedSampler,
        SamplerType.NSGAII: optuna.samplers.NSGAIISampler,
        SamplerType.MOTPE: optuna.samplers.MOTPESampler,
    }

    @staticmethod
    def get_other_args(kwargs, kwspaces):
        """Get key-word arguments which are not search spaces."""
        return{k: kwargs[k] for k in set(kwargs) - set(kwspaces)}

    @staticmethod
    def get_hpo_config(trial, configspace):
        """Get hyper parameter suggestions from search space settings."""
        # TODO better ways to map ConfigSpace to optuna spaces
        # fix order of hyperparams in configspace.
        hp_ordering = configspace.get_hyperparameter_names()
        config = {}
        for hp in hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_prefix = hp_obj.meta.setdefault('prefix', None)
            hp_name = hp_prefix + ':' + hp if hp_prefix else hp
            hp_type = str(type(hp_obj)).lower()  # type of hyperparam
            if 'integer' in hp_type:
                hp_dimension = trial.suggest_int(
                    name=hp_name, low=int(hp_obj.lower), high=int(hp_obj.upper))
            elif 'float' in hp_type:
                if hp_obj.log:  # log10-scale hyperparmeter
                    hp_dimension = trial.suggest_loguniform(
                        name=hp_name, low=float(hp_obj.lower), high=float(hp_obj.upper))
                else:
                    hp_dimension = trial.suggest_float(
                        name=hp_name, low=float(hp_obj.lower), high=float(hp_obj.upper))
            elif 'categorical' in hp_type:
                hp_dimension = trial.suggest_categorical(
                    name=hp_name, choices=hp_obj.choices)
            elif 'ordinal' in hp_type:
                hp_dimension = trial.suggest_categorical(
                    name=hp_name, choices=hp_obj.sequence)
            else:
                raise ValueError("unknown hyperparameter type: %s" % hp)
            config[hp_name] = hp_dimension
        return config

    @staticmethod
    def instantiate(trial, lazyobj):
        """Instantiate a lazyobject from a trial's sampled param set."""
        config = OptunaBackend.gen_config(trial, lazyobj)
        return lazyobj.sample(**config)

    @staticmethod
    def gen_config(trial, automl_obj):
        """Generate the param config from a trial's sampled param set."""
        configspace = automl_obj.cs
        config = OptunaBackend.get_hpo_config(trial, configspace)
        other_kwargs = OptunaBackend.get_other_args(
            automl_obj.kwargs, automl_obj.kwspaces)
        config.update(other_kwargs)
        return config

    @staticmethod
    def create_sampler(sampler_type, kwargs):
        """Create a hyperparameter sampler by type."""
        sampler_class = OptunaBackend.sampler_map.get(sampler_type)
        return sampler_class(**kwargs)

    @staticmethod
    def create_pruner(pruner_type, kwargs):
        """Create a pruner by type."""
        pruner_class = OptunaBackend.pruner_map.get(pruner_type)
        return pruner_class(**kwargs)

    @staticmethod
    def create_study(**kwargs):
        """Create a study to drive the hyperparameter search."""
        return optuna.create_study(**kwargs)
