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

from .space import Bool, Int, Categorical, Real


class OptunaBackend(object):

    @staticmethod
    def get_other_args(kwargs, kwspaces):
        return{ k : kwargs[k] for k in set(kwargs) - set(kwspaces) }

    @staticmethod
    def get_hpo_config(trial, configspace):
        # TODO better ways to map ConfigSpace to optuna spaces
        hp_ordering = configspace.get_hyperparameter_names() # fix order of hyperparams in configspace.
        config={}
        for hp in hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower() # type of hyperparam
            if 'integer' in hp_type:
                hp_dimension = trial.suggest_int(name=hp, low=int(hp_obj.lower), high=int(hp_obj.upper))
            elif 'float' in hp_type:
                if hp_obj.log: # log10-scale hyperparmeter
                    hp_dimension = trial.suggest_loguniform(name=hp, low=float(hp_obj.lower), high=float(hp_obj.upper))
                else:
                    hp_dimension = trial.suggest_float(name=hp, low=float(hp_obj.lower), high=float(hp_obj.upper))
            elif 'categorical' in hp_type:
                hp_dimension = trial.suggest_categorical(name=hp, choices=hp_obj.choices)
            elif 'ordinal' in hp_type:
                hp_dimension = trial.suggest_categorical(name=hp, choices = hp_obj.sequence)
            else:
                raise ValueError("unknown hyperparameter type: %s" % hp)
            config[hp] = hp_dimension
        return config


    @staticmethod
    def instantiate(trial, lazyobj):
        config = OptunaBackend.gen_config(trial, lazyobj)
        return lazyobj.sample(**config)

    @staticmethod
    def gen_config(trial, automl_obj):
        configspace = automl_obj.cs
        config = OptunaBackend.get_hpo_config(trial, configspace)
        other_kwargs=OptunaBackend.get_other_args(automl_obj.kwargs, automl_obj.kwspaces)
        config.update(other_kwargs)
        return config
