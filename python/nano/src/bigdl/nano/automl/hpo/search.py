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

# This file is adapted from https://github.com/awslabs/autogluon/
# blob/0.3.1/core/src/autogluon/core/space.py
# Copyright The AutoGluon project at https://github.com/awslabs/autogluon
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License at
# https://github.com/awslabs/autogluon/blob/master/LICENSE


import optuna
from bigdl.nano.utils.common import invalidInputError


def _filter_tuner_args(kwargs, tuner_keys):
    return {k: v for k, v in kwargs.items() if k in tuner_keys}


def _search_summary(study):
    """
    Print statistics of trials and retrieve the summary for further analysis.

    :param study: the hpo study object
    :return : the summary object (current we return the study directly, so that
        it allows better flexiblity to do visualization and futher analysis)
    """
    if study is not None:
        print("Number of finished trials: {}".format(len(study.trials)))
        best = study.best_trial
        print("Best trial:")
        print("  Value: {}".format(best.value))
        print("  Params: ")
        for key, value in best.params.items():
            print("    {}: {}".format(key, value))
        return study
        # return self.study.trials_dataframe(attrs=("number", "value", "params", "state"))
    else:
        print("Seems you have not done any tuning yet.  \
                Call tune and then call tune_summary to get the statistics.")


def _end_search(study, model_builder, use_trial_id=-1):
    """
    Put an end to tuning.

    Use the specified trial or best trial to init and compile the base model.

    :param study: the hpo study object.
    :param model_builder: the function to build the model.
    :param use_trial_id: int(optional) params of which trial to be used. Defaults to -1.
    :throw ValueError: if study is None.
    :return : the built model with best or specified trial hyperparams.
    """
    if study is None:
        trial = optuna.trial.FixedTrial({})
    elif use_trial_id == -1:
        trial = study.best_trial
    else:
        trial = study.trials[use_trial_id]

    _lazymodel = model_builder(trial)
    # TODO Next step: support retrive saved model instead of retrain from hparams
    return _lazymodel


def _create_study(resume, create_kwargs, backend):

    if not resume:
        load_if_exists = False
        print("Starting a new tuning")
    else:
        load_if_exists = True
        print("Resume the last tuning...")
    create_kwargs['load_if_exists'] = load_if_exists
    # create study
    return backend.create_study(**create_kwargs)


def _check_search_args(search_args, legal_keys):
    search_arg_keys = set(search_args.keys())
    allkeys = set().union(*legal_keys)
    illegal_args = search_arg_keys.difference(allkeys)
    if len(illegal_args) > 0:
        invalidInputError(False,
                          'Invalid Arguments found for \'search\':')


def _prepare_args(kwargs,
                  create_keys,
                  run_keys,
                  fit_keys,
                  backend):

    create_kwargs = _filter_tuner_args(kwargs, create_keys)
    run_kwargs = _filter_tuner_args(kwargs, run_keys)
    fit_kwargs = _filter_tuner_args(kwargs, fit_keys)
    # prepare sampler and pruner args
    sampler_type = create_kwargs.get('sampler', None)
    if sampler_type:
        sampler_args = create_kwargs.get('sampler_kwargs', None)
        sampler = backend.create_sampler(sampler_type, sampler_args)
        create_kwargs['sampler'] = sampler
        create_kwargs.pop('sampler_kwargs', None)

    pruner_type = create_kwargs.get('pruner', None)
    if pruner_type:
        pruner_args = create_kwargs.get('pruner_kwargs', {})
        pruner = backend.create_pruner(pruner_type, pruner_args)
        create_kwargs['pruner'] = pruner
        create_kwargs.pop('pruner_kwargs', None)

    # renamed callbacks to tune_callbacks to avoid conflict with fit param
    run_kwargs['callbacks'] = run_kwargs.get('tune_callbacks', None)
    run_kwargs.pop('tune_callbacks', None)
    run_kwargs['show_progress_bar'] = False

    return create_kwargs, run_kwargs, fit_kwargs


def _validate_args(search_kwargs,
                   target_metric,
                   legal_keys):
    _check_search_args(
        search_args=search_kwargs,
        legal_keys=legal_keys)

    direction = search_kwargs.get('direction', None)
    directions = search_kwargs.get('directions', None)
    _check_optimize_direction(
        direction=direction,
        directions=directions,
        metric=target_metric)


def _strip_val_prefix(metric):
    if metric.startswith('val_'):
        metric = metric[len('val_'):]
    return metric


def _check_optimize_direction(direction, directions, metric):
    # TODO check common metrics and corresponding directions
    if (isinstance(metric, list) or isinstance(metric, tuple)) and len(metric) > 1:
        # multi-objective search
        invalidInputError(directions is not None and len(directions) == len(metric),
                          "In multi-objective optimization, you must explicitly specify "
                          "the direction for each metric")
    else:
        if not direction and not directions:
            direction = 'minimize'
        max_metrics = ['accuracy', 'auc', 'acc']
        min_metrics = ['loss', 'mae', 'mse']
        stripped_metric = _strip_val_prefix(metric).lower()
        if stripped_metric in max_metrics:
            if direction != 'maximize':
                invalidInputError(False,
                                  'should use maximize direction for optmize')
        elif stripped_metric in min_metrics:
            if direction != 'minimize':
                invalidInputError(False,
                                  'should use minimize direction for optmize')
