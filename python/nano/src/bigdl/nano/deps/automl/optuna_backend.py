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

from bigdl.nano.automl.hpo.backend import PrunerType, SamplerType
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.automl.hpo.space import SimpleSpace, NestedSpace, AutoObject
from bigdl.nano.automl.hpo.space import (
    AutoObject, Space, SingleParam,
    _get_hp_prefix)
import optuna


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

    SPLITTER = u':'  # For splitting prefix and names of hyperparam

    @staticmethod
    def get_other_args(kwargs, kwspaces):
        """Get key-word arguments which are not search spaces."""
        return{k: kwargs[k] for k in set(kwargs) - set(kwspaces)}

    @staticmethod
    def _sample_space(trial, hp_name, hp_obj):
        hp_type = str(type(hp_obj)).lower()  # type of hyperparam
        if 'integer' in hp_type or 'float' in hp_type or \
                'categorical' in hp_type or 'ordinal' in hp_type:
            try:
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
            except (RuntimeError):
                # TODO ValueErrors might be throw due to other reasons.
                invalidInputError(False,
                                  "If you set search space in model, "
                                  "you must call model.search before model.fit.")
        else:
            invalidInputError(False,
                              "unknown hyperparameter type %s for param %s" %
                              (hp_type, hp_name))
        return hp_dimension

    @staticmethod
    def get_hpo_config(trial, configspace):
        """Get hyper parameter suggestions from search space settings."""
        # TODO better ways to map ConfigSpace to optuna spaces
        hp_ordering = configspace.get_hyperparameter_names()
        config = {}
        # hp_prefix = _get_cs_prefix(configspace)
        for hp_name in hp_ordering:
            hp = configspace.get_hyperparameter(hp_name)
            # TODO generate meaningful prefix for user in AutoObj
            hp_prefix = _get_hp_prefix(hp)
            optuna_hp_name = OptunaBackend._format_hp_name(hp_prefix, hp_name)
            hp_dimension = OptunaBackend._sample_space(trial, optuna_hp_name, hp)
            config[hp_name] = hp_dimension
        return config

    @staticmethod
    def _format_hp_name(prefix, hp_name):
        if prefix:
            return "{}{}{}".format(
                prefix, OptunaBackend.SPLITTER, hp_name)
        else:
            return hp_name

    @staticmethod
    def instantiate_param(trial, kwargs, arg_name):
        """
        Instantiate auto objects in kwargs with trial params at runtime.

        Note the params are replaced IN-PLACE
        """
        # instantiate auto objects in runtime params a
        v = kwargs.get(arg_name, None)

        if not v:
            return kwargs

        if not isinstance(v, Space):
            value = v
        elif isinstance(v, AutoObject):
            value = OptunaBackend.instantiate(trial, v)
        else:
            pobj = SingleParam(arg_name, v)
            config = OptunaBackend.get_hpo_config(trial, pobj.cs)
            value = pobj.sample(**config)

        kwargs[arg_name] = value
        return kwargs

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
        return sampler_class(kwargs)

    @staticmethod
    def create_pruner(pruner_type, kwargs):
        """Create a pruner by type."""
        pruner_class = OptunaBackend.pruner_map.get(pruner_type)
        return pruner_class(**kwargs)

    @staticmethod
    def create_study(**kwargs):
        """Create a study to drive the hyperparameter search."""
        return optuna.create_study(**kwargs)
