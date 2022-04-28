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


def _filter_tuner_args(kwargs, tuner_keys):
    return {k: v for k, v in kwargs.items() if k in tuner_keys}


def _search_summary(study):
    """
    Print statistics of trials and retrieve the summary for further analysis.

    :param study: the optuna study object
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

    :param study: the optuna study object.
    :param model_builder: the function to build the model.
    :param use_trial_id: int(optional) params of which trial to be used. Defaults to -1.
    :raises ValueError: if study is None.
    :return : the built model with best or specified trial hyperparams.
    """
    if study is None:
        raise ValueError("study is None.   \
                            Please call search before calling end_search. ")
    if use_trial_id == -1:
        trial = study.best_trial
    else:
        trial = study.trials[use_trial_id]

    _lazymodel = model_builder(trial)
    # TODO Next step: support retrive saved model instead of retrain from hparams
    return _lazymodel


def _check_search_args(search_args, legal_keys):
    search_arg_keys = set(search_args.keys())
    allkeys = set().union(*legal_keys)
    illegal_args = search_arg_keys.difference(allkeys)
    if len(illegal_args) > 0:
        raise ValueError('Invalid Arguments found for \'search\':',
                         ', '.join(illegal_args))


def _strip_val_prefix(metric):
    if metric.startswith('val_'):
        metric = metric[len('val_'):]
    return metric


def _check_optimize_direction(direction, directions, metric):
    # TODO check common metrics and corresponding directions
    if directions:
        # TODO we don't check for multiobjective cases
        return
    if not direction and not directions:
        direction = 'minimize'
    max_metrics = ['accuracy', 'auc', 'acc']
    min_metrics = ['loss', 'mae', 'mse']
    stripped_metric = _strip_val_prefix(metric).lower()
    if stripped_metric in max_metrics:
        if direction != 'maximize':
            raise ValueError('metric', metric,
                             'should use maximize direction for optmize')
    elif stripped_metric in min_metrics:
        if direction != 'minimize':
            raise ValueError('metric', metric,
                             'should use minimize direction for optmize')
