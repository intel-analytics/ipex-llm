import copy
import logging
import argparse
import functools
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from .backend import OptunaBackend

from .space import *
from .space import _add_hp, _add_cs, _rm_hp, _strip_config_space, SPLITTER

from .mixin import HPOMixin

from .callgraph import update_callgraph
from bigdl.nano.automl.utils import EasyDict as ezdict
from bigdl.nano.automl.utils.lazyutils import proxy_methods

__all__ = ['args', 'obj', 'func', 'model', 'sample_config']

logger = logging.getLogger(__name__)


def sample_config(args, config):
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
        #from .reporter import FakeReporter
        #if 'reporter' not in kwargs:
        #    logger.debug('Creating FakeReporter for test purpose.')
        #    kwargs['reporter'] = FakeReporter()

        output = self.f(args, **kwargs)
        #logger.debug('Reporter Done!')
        #kwargs['reporter'](done=True)
        return output

    def register_args(self, default={}, **kwvars):
        if isinstance(default, (argparse.Namespace, argparse.ArgumentParser)):
            default = vars(default)
        self.kwvars = {}
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        """For searcher support ConfigSpace
        """
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
        cs = CS.ConfigurationSpace()
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
        """For RL searcher/controller
        """
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
    """Decorator for a Python training script that registers its arguments as hyperparameters.
       Each hyperparameter takes fixed value or is a searchable space, and the arguments may either be:
       built-in Python objects (e.g. floats, strings, lists, etc.), AutoObject (see :func:`hpo.obj`),
       or hpo search spaces (see :class:`hpo.space.Int`, :class:`hpo.space.Real`, etc.).

    Examples
    --------
    >>>
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
    """Decorator for a function that registers its arguments as hyperparameters.
       Each hyperparameter may take a fixed value or be a searchable space (hpo.space).

    Returns
    -------
    Instance of :class:`hpo.space.AutoObject`:
        A lazily initialized object, which allows for distributed training.

    Examples
    --------
    >>>
    """
    def _automl_kwargs_func(**kwvars):
        def registered_func(func):
            kwspaces = OrderedDict()
            @functools.wraps(func)
            def wrapper_call(*args, **kwargs):
                _kwvars = copy.deepcopy(kwvars)
                _kwvars.update(kwargs)
                for k, v in _kwvars.items():
                    if isinstance(v, NestedSpace):
                        kwspaces[k] = v
                        kwargs[k] = v
                    elif isinstance(v, Space):
                        kwspaces[k] = v
                        hp = v.get_hp(name=k)
                        kwargs[k] = hp.default_value
                    else:
                        kwargs[k] = v
                return func(*args, **kwargs)
            wrapper_call.kwspaces = kwspaces
            return wrapper_call
        return registered_func

    def registered_func(func):
        class automlobject(AutoObject):
            @_automl_kwargs_func(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(automlobject.kwspaces)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]

                return self.func(*self.args, **kwargs)

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            _kwvars = copy.deepcopy(kwvars)
            _kwvars.update(kwargs)
            agobj = automlobject(*args, **kwargs)
            agobj.kwvars = _kwvars
            return agobj
        return wrapper_call
    return registered_func

def obj(**kwvars):
    """Decorator for a Python class that registers its arguments as hyperparameters.
       Each hyperparameter may take a fixed value or be a searchable space (hpo.space).

    Returns
    -------
    Instance of :class:`hpo.space.AutoObject`:
        A lazily initialized object, which allows distributed training.

    Examples
    --------
    >>>
    """
    def _automl_kwargs_obj(**kwvars):
        def registered_func(func):
            kwspaces = OrderedDict()
            @functools.wraps(func)
            def wrapper_call(*args, **kwargs):
                kwvars.update(kwargs)
                for k, v in kwvars.items():
                    if isinstance(v, NestedSpace):
                        kwspaces[k] = v
                        kwargs[k] = v
                    elif isinstance(v, Space):
                        kwspaces[k] = v
                        hp = v.get_hp(name=k)
                        kwargs[k] = hp.default_value
                    else:
                        kwargs[k] = v
                return func(*args, **kwargs)
            wrapper_call.kwspaces = kwspaces
            wrapper_call.kwvars = kwvars
            return wrapper_call
        return registered_func

    def registered_class(Cls):
        class automlobject(AutoObject):
            @_automl_kwargs_obj(**kwvars)
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self._inited = False
                self._callgraph = None #keep a reference to the call graph

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(automlobject.kwspaces)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]

                args = self.args
                return Cls(*args, **kwargs)

            def __repr__(self):
                return 'AutoObject -- ' + Cls.__name__

            def __call__(self, *args, **kwargs):
                # this is to handle functional API of layers
                self._call_args = args
                self._call_kwargs = kwargs
                #get the inputs tensor argument
                if len(args) == 0:
                    inputs = kwargs['inputs']
                else:
                    inputs = args[0]
                self._callgraph = update_callgraph(inputs, self)
                return self

        automlobject.kwvars = automlobject.__init__.kwvars
        automlobject.__doc__ = Cls.__doc__
        automlobject.__name__ = Cls.__name__
        return automlobject

    return registered_class



def model(**kwvars):
    def registered_class(Cls):
        objCls = obj(**kwvars)(Cls)
        @proxy_methods
        class AutomlModel(HPOMixin, Cls):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._lazyobj =objCls(**kwargs)
                #generate a default config for the super class
                default_config = self._lazyobj.cs.get_default_configuration().get_dictionary()
                super_kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(self._lazyobj.kwspaces)
                for k, v in super_kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(default_config, prefix=k)
                        super_kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in default_config:
                        super_kwargs[k] = default_config[k]
                super().__init__(**super_kwargs)


            def _model_build(self, trial):
                #override _model_build to build
                # the model directly instead of using
                # modeld_init and model_compile
                model = OptunaBackend.instantiate(trial,self._lazyobj)
                self._model_compile(model, trial)
                return model

        return AutomlModel

    return registered_class

