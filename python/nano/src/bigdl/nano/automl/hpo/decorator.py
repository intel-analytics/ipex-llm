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
# blob/v0.3.1/core/src/autogluon/core/decorator.py
# Copyright The AutoGluon project at https://github.com/awslabs/autogluon
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License at
# https://github.com/awslabs/autogluon/blob/master/LICENSE


import copy
import logging
import argparse
import functools
from collections import OrderedDict
import numpy as np
import multiprocessing as mp


from .space import *
from .space import _add_hp, _add_cs, _rm_hp, _strip_config_space, SPLITTER
from .callgraph import CallCache

from bigdl.nano.automl.utils import EasyDict as ezdict
from bigdl.nano.automl.utils import proxy_methods
from bigdl.nano.automl.hpo.backend import create_hpo_backend
from bigdl.nano.deps.automl.hpo_api import create_configuration_space

__all__ = ['args', 'obj', 'func', 'tfmodel', 'plmodel', 'sample_config']

logger = logging.getLogger(__name__)


def sample_config(args, config):
    """
    Sample a set of hyperparams from given config.

    :param args: The arguments with possbile search space params.
    :param config: a combinition of hyperparams (e.g. obtained in each trial)
    :return: a sampled set of hyperparams.
    """
    args = copy.deepcopy(args)
    striped_keys = [k.split(SPLITTER)[0] for k in config.keys()]
    if isinstance(args, (argparse.Namespace, argparse.ArgumentParser)):
        args_dict = vars(args)
    else:
        args_dict = args
    for k, v in args_dict.items():
        # handle different type of configurations
        if k in striped_keys:
            if isinstance(v, NestedSpace):
                sub_config = _strip_config_space(config, prefix=k)
                args_dict[k] = v.sample(**sub_config)
            else:
                if SPLITTER in k:
                    continue
                args_dict[k] = config[k]
        elif isinstance(v, AutoObject):
            args_dict[k] = v.init()
    return args


class _automl_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()

    def __init__(self, f):
        self.f = f
        self.args = ezdict()
        functools.update_wrapper(self, f)

    def __call__(self, args, config={}, **kwargs):
        new_config = copy.deepcopy(config)
        self._rand_seed()
        args = sample_config(args, new_config)
        # from .reporter import FakeReporter
        # if 'reporter' not in kwargs:
        #    logger.debug('Creating FakeReporter for test purpose.')
        #    kwargs['reporter'] = FakeReporter()

        output = self.f(args, **kwargs)
        # logger.debug('Reporter Done!')
        # kwargs['reporter'](done=True)
        return output

    def register_args(self, default={}, **kwvars):
        if isinstance(default, (argparse.Namespace, argparse.ArgumentParser)):
            default = vars(default)
        self.kwvars = {}
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        # for searcher support ConfigSpace
        self.kwvars.update(kwargs)
        for k, v in self.kwvars.items():
            if isinstance(v, (NestedSpace)):
                self.args.update({k: v})
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                self.args.update({k: hp.default_value})
            else:
                self.args.update({k: v})

    @property
    def cs(self):
        cs = create_configuration_space()
        for k, v in self.kwvars.items():
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, k)
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                _add_hp(cs, hp)
            else:
                _rm_hp(cs, k)
        return cs

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for k, v in self.kwvars.items():
            if isinstance(v, NestedSpace):
                if isinstance(v, Categorical):
                    kw_spaces['{}{}choice'.format(k, SPLITTER)] = v
                for sub_k, sub_v in v.kwspaces.items():
                    new_k = '{}{}{}'.format(k, SPLITTER, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(v, Space):
                kw_spaces[k] = v
        return kw_spaces

    def _rand_seed(self):
        _automl_method.SEED.value += 1
        np.random.seed(_automl_method.SEED.value)

    def __repr__(self):
        return repr(self.f)


def args(default=None, **kwvars):
    """
    Decorator for a Python script that registers its arguments as hyperparameters.

    Each hyperparameter may take a fixed value or be a searchable space
    (e.g. hpo.space.Int, hpo.space.Categorical).

    :param default: a dictionary of hyperparameter default values, defaults to None
    :return : a wrapped function.
    """
    if default is None:
        default = dict()
    kwvars['_default_config'] = default

    def registered_func(func):
        @_automl_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        default = kwvars['_default_config']
        wrapper_call.register_args(default=default, **kwvars)
        return wrapper_call

    return registered_func


def func(**kwvars):
    """
    Decorator for a function that registers its arguments as hyperparameters.

    Used for functions like: tf.cast, tf.keras.Input, tf.keras.activations.*.
    Each hyperparameter may take a fixed value or be a searchable space
    (e.g. hpo.space.Int, hpo.space.Categorical).

    :return: an AutoFunc object. The function body is not immediately executed at the time
             of user invocation. The actual execution is delayed until AutoFunc.sample()
             is called (usually in each trial).
    """
    from .callgraph import CallCache, CALLTYPE

    # def _automl_kwargs_func(**kwvars):
    #     def registered_func(func):
    #         kwspaces = OrderedDict()

    #         @functools.wraps(func)
    #         def wrapper_call(*args, **kwargs):
    #             _kwvars = copy.deepcopy(kwvars)
    #             _kwvars.update(kwargs)
    #             for k, v in _kwvars.items():
    #                 if isinstance(v, NestedSpace):
    #                     kwspaces[k] = v
    #                     kwargs[k] = v
    #                 elif isinstance(v, Space):
    #                     kwspaces[k] = v
    #                     hp = v.get_hp(name=k)
    #                     kwargs[k] = hp.default_value
    #                 else:
    #                     kwargs[k] = v
    #             return func(*args, **kwargs)
    #         wrapper_call.kwspaces = kwspaces
    #         return wrapper_call
    #     return registered_func

    def registered_func(func):
        class AutoSlice(AutoObject):
            def __init__(self, source, slice_arguments):
                self.source = source
                self.slice_arguments = slice_arguments
                self._callgraph = CallCache.update(
                    (self.source, self.slice_arguments),
                    self,
                    ctype=CALLTYPE.FUNC_SLICE)

            def sample(self, **config):
                slice_args, slice_kwargs = self.slice_arguments
                return self.source.__getitem__(*slice_args, **slice_kwargs)

            @property
            def kwspaces(self):
                return {}

            def __repr__(self):
                return 'AutoSlice -- [] ' + str(id(self))

        class AutoFunc(AutoObject):
            # @_automl_kwargs_func(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self._inited = False

                self.slice_args = None
                self.slice_kwargs = None

                self.kwspaces_ = OrderedDict()
                self.kwvars = dict()
                self._update_kw()

                self._callgraph = None  # keep a reference to the call graph
                self._callgraph = CallCache.update(
                    (self.args, self.kwargs),
                    self,
                    ctype=CALLTYPE.FUNC_CALL)

            def __getitem__(self, *args, **kwargs):
                self.slice_args = args
                self.slice_kwargs = kwargs
                return AutoSlice(self, (self.slice_args, self.slice_kwargs))

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                # kwspaces = copy.deepcopy(AutoFunc.kwspaces)
                kwspaces = copy.deepcopy(self.kwspaces_)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]

                return self.func(*self.args, **kwargs)

            @property
            def kwspaces(self):
                return self.kwspaces_

            def _update_kw(self):
                self.kwvars.update(self.kwargs)
                for k, v in self.kwvars.items():
                    if isinstance(v, NestedSpace):
                        self.kwspaces_[k] = v
                        self.kwargs[k] = v
                    elif isinstance(v, Space):
                        self.kwspaces_[k] = v
                        hp = v.get_hp(name=k)
                        self.kwargs[k] = hp.default_value
                    else:
                        self.kwargs[k] = v

            def __repr__(self):
                return 'AutoFunc -- ' + self.func.__name__ + ' ' + str(id(self))

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            _kwvars = copy.deepcopy(kwvars)
            _kwvars.update(kwargs)
            autoobj = AutoFunc(*args, **kwargs)
            autoobj.kwvars = _kwvars
            return autoobj
        return wrapper_call
    return registered_func


def obj(**kwvars):
    """
    Decorator for a class that registers its arguments as hyperparameters.

    Used for classes like: tf.keras.layers.*
    Each hyperparameter may take a fixed value or be a searchable space
    (e.g. hpo.space.Int, hpo.space.Categorical).


    :return: an AutoCls object. The instantiation of the class object is delayed
             until AutoCls.sample() is called (usually in each trial).

    """
    # def _automl_kwargs_obj(**kwvars):
    #     def registered_func(func):
    #         kwspaces = OrderedDict()
    #         @functools.wraps(func)
    #         def wrapper_call(*args, **kwargs):
    #             kwvars.update(kwargs)
    #             for k, v in kwvars.items():
    #                 if isinstance(v, NestedSpace):
    #                     kwspaces[k] = v
    #                     kwargs[k] = v
    #                 elif isinstance(v, Space):
    #                     kwspaces[k] = v
    #                     hp = v.get_hp(name=k)
    #                     kwargs[k] = hp.default_value
    #                 else:
    #                     kwargs[k] = v
    #             return func(*args, **kwargs)
    #         wrapper_call.kwspaces = kwspaces
    #         wrapper_call.kwvars = kwvars
    #         return wrapper_call
    #     return registered_func

    def registered_class(Cls):
        class AutoCls(AutoObject):
            # @_automl_kwargs_obj(**kwvars)
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self._inited = False
                self.kwspaces_ = OrderedDict()
                self.kwvars = dict()
                self.sampler_kwargs = dict()
                self._update_kw()
                self._callgraph = None  # keep a reference to the call graph

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                # kwspaces = copy.deepcopy(automlobject.kwspaces)
                kwspaces = copy.deepcopy(self.kwspaces_)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]

                args = self.args
                return Cls(*args, **kwargs)

            @property
            def kwspaces(self):
                return self.kwspaces_

            def _update_kw(self):
                self.kwvars.update(self.kwargs)
                for k, v in self.kwvars.items():
                    if isinstance(v, NestedSpace):
                        self.kwspaces_[k] = v
                        self.kwargs[k] = v
                        for hp in v.cs.get_hyperparameters():
                            new_parameter = copy.deepcopy(hp)
                            new_parameter.name = "{}{}{}".format(
                                k, SPLITTER, new_parameter.name)
                            # further add the sub_cs prefix onto the param
                            name = new_parameter.name
                            if hasattr(new_parameter, "choices"):
                                choices = tuple(new_parameter.choices)
                                self.sampler_kwargs[name] = choices
                    elif isinstance(v, Space):
                        self.kwspaces_[k] = v
                        hp = v.get_hp(name=k)
                        self.kwargs[k] = hp.default_value
                    else:
                        self.kwargs[k] = v

            def __repr__(self):
                return 'AutoCls -- ' + Cls.__name__ + str(id(self))

            def __call__(self, *args, **kwargs):

                # super.__call__(*args, **kwargs)
                # this is to handle functional API of layers
                self._call_args = args
                self._call_kwargs = kwargs
                # get the inputs tensor argument
                if len(args) == 0:
                    inputs = kwargs['inputs']
                else:
                    inputs = args[0]
                self._callgraph = CallCache.update(inputs, self)
                return self

        # automlobject.kwvars = automlobject.__init__.kwvars
        AutoCls.__doc__ = Cls.__doc__
        AutoCls.__name__ = Cls.__name__
        return AutoCls

    return registered_class


def tfmodel(**kwvars):
    """
    Decorator for a Tensorflow model that registers its arguments as hyperparameters.

    Used for custom keras model subclassing from tf.keras.Model.
    Each hyperparameter may take a fixed value or be a searchable space
    (e.g. hpo.space.Int, hpo.space.Categorical).

    :return: a TFAutoMdl object. The instantiation of the class object is delayed
             until TFAutoMdl.sample() is called (usually in each trial). The difference
             between a TFAutoMdl and a AutoCls is TFAutoMdl has search and search_summary
             methods.

    """
    from bigdl.nano.automl.tf.mixin import HPOMixin

    def registered_class(Cls):
        objCls = obj(**kwvars)(Cls)

        @proxy_methods
        class TFAutoMdl(HPOMixin, Cls):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._lazyobj = objCls(**kwargs)
                # generate a default config for the super class
                default_config = self._lazyobj.cs.get_default_configuration().get_dictionary()
                super_kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(self._lazyobj.kwspaces)
                for k, v in super_kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(
                            default_config, prefix=k)
                        super_kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in default_config:
                        super_kwargs[k] = default_config[k]
                super().__init__(**super_kwargs)

            def __repr__(self):
                return 'TFAutoMdl -- ' + Cls.__name__

            def _model_build(self, trial):
                # override _model_build to build
                # the model directly instead of using
                # modeld_init and model_compile
                model = self.backend.instantiate(trial, self._lazyobj)
                self._model_compile(model, trial)
                return model

            def _get_model_builder_args(self):
                return {'lazyobj': self._lazyobj,
                        'compile_args': self.compile_args,
                        'compile_kwargs': self.compile_kwargs,
                        'backend': self.backend}

            @staticmethod
            def _get_model_builder(lazyobj,
                                   compile_args,
                                   compile_kwargs,
                                   backend):

                def model_builder(trial):
                    model = backend.instantiate(trial, lazyobj)
                    # self._model_compile(model, trial)
                    # instantiate optimizers if it is autoobj
                    optimizer = compile_kwargs.get('optimizer', None)
                    if optimizer and isinstance(optimizer, AutoObject):
                        optimizer = backend.instantiate(trial, optimizer)
                        compile_kwargs['optimizer'] = optimizer
                    model.compile(*compile_args, **compile_kwargs)
                    return model
                return model_builder

        return TFAutoMdl

    return registered_class


def plmodel(**kwvars):
    """
    Decorator for a Custom PyTorch model that registers its arguments as hyperparameters.

    Used for custom pytorch model subclassing from pytorch_lightning.LightningModule.
    Each hyperparameter may take a fixed value or be a searchable space
    (e.g. hpo.space.Int, hpo.space.Categorical).

    :return: a PlAutoMdl object. The instantiation of the class object is delayed
             until PlAutoMdl.sample() is called (usually in each trial). Unlike TFAutoMdl,
             PlAutoMdl, does not have search and search_summary methods.
    >>>
    """
    def registered_class(Cls):
        objCls = obj(**kwvars)(Cls)

        class PLAutoMdl(Cls):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._lazyobj = objCls(**kwargs)
                # generate a default config for the super class
                default_config = self._lazyobj.cs.get_default_configuration().get_dictionary()
                super_kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(self._lazyobj.kwspaces)
                for k, v in super_kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(
                            default_config, prefix=k)
                        super_kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in default_config:
                        super_kwargs[k] = default_config[k]
                super().__init__(**super_kwargs)
                self.backend = create_hpo_backend()

            def __repr__(self):
                return 'PlAutoMdl -- ' + Cls.__name__

            def _model_build(self, trial):
                # override _model_build to build
                # the model directly instead of using
                # modeld_init and model_compile
                model = self.backend.instantiate(trial, self._lazyobj)
                # self._model_compile(model, trial)
                return model

        return PLAutoMdl

    return registered_class
