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


import copy
from collections import OrderedDict
from bigdl.nano.automl.utils import EasyDict
from bigdl.nano.deps.automl.hpo_api import create_configuration_space
from bigdl.nano.deps.automl.hpo_api import (
    create_uniform_float_hp, create_uniform_int_hp, create_categorical_hp)
from bigdl.nano.utils.common import invalidInputError


__all__ = ['Space', 'NestedSpace', 'AutoObject', 'List', 'Dict',
           'Categorical', 'Real', 'Int', 'Bool']

SPLITTER = u'▁'  # Use U+2581 as the special symbol for splitting the space


class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Space(object):
    """
    Base search space.

    Describing set of candidate values for hyperparameters.
    """

    pass


class SimpleSpace(Space):
    """Non-nested search space. (i.e. a single simple hyperparameter)."""

    def __init__(self, prefix=None):
        """Init prefix for the param space."""
        self._prefix = prefix

    def __repr__(self):
        reprstr = self.__class__.__name__
        if hasattr(self, 'lower') and hasattr(self, 'upper'):
            reprstr += ': lower={}, upper={}'.format(self.lower, self.upper)
        if hasattr(self, 'value'):
            reprstr += ': value={}'.format(self.value)
        return reprstr

    def get_hp(self, name):
        """Fetch particular hyperparameter based on its name."""
        invalidInputError(False, "not implement get_hp for SimpleSpace")

    @property
    def hp(self):
        """Return hyperparameter corresponding to this search space."""
        return self.get_hp(name='')

    @property
    def prefix(self):
        """Get the prefix."""
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        """Set the prefix in HP."""
        self._prefix = value

    @property
    def default(self):
        """Return default value of hyperparameter corresponding to\
           this search space. This value is tried first during\
           hyperparameter optimization."""
        default = self._default if self._default else self.hp.default_value
        return default

    @default.setter
    def default(self, value):
        """Set default value for hyperparameter corresponding to\
           this search space. The default value is always tried\
           in the first trial of HPO."""
        self._default = value

    @property
    def rand(self):
        """Return randomly sampled (but valid) value from this search space."""
        cs = _new_cs(self.prefix)
        cs.add_hyperparameter(self.hp)
        return cs.sample_configuration().get_dictionary()['']


class NestedSpace(Space):
    """Nested hyperparameter search space, which is a search space that\
       itself contains multiple search spaces."""

    def __init__(self, prefix=None):
        """Init prefix for the param space."""
        self._prefix = prefix

    def sample(self, **config):
        """Sample a configuration from this search space."""
        pass

    @property
    def cs(self):
        """`ConfigSpace` representation of this search space."""
        invalidInputError(False, "not implement cs for NestedSpace")

    @property
    def prefix(self):
        """Get the prefix."""
        return self._prefix if hasattr(self, '_prefix') else None

    @prefix.setter
    def prefix(self, value):
        """Set the prefix in HP."""
        self._prefix = value

    @property
    def kwspaces(self):
        """`OrderedDict` representation of this search space."""
        invalidInputError(False, "not implement kwspaces for NestedSpace")

    @property
    def default(self):
        """Return default value for hyperparameter corresponding to\
           this search space. The default value is always tried\
           in the first trial of HPO."""
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

    @property
    def rand(self):
        """Randomly sample configuration from this nested search space."""
        config = self.cs.sample_configuration().get_dictionary()
        return self.sample(**config)


class AutoObject(NestedSpace):
    """
    Searchable objects.

    Created by decorating a custom Python class or function using the
    `hpo.obj` or `hpo.func` decorators.
    """

    def __call__(self, *args, **kwargs):
        """Convenience method for interacting with AutoObject."""
        if not self._inited:
            self._inited = True
            self._instance = self.init()
        return self._instance.__call__(*args, **kwargs)

    def init(self):
        """Instantiate an actual instance of this `AutoObject`.\
            In order to interact with such an `object`,\
            you must always first call: `object.init()`."""
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

    @property
    def cs(self):
        """`ConfigSpace` representation of this search space."""
        cs = _new_cs(self.prefix)
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
        """`OrderedDict` representation of this search space."""
        invalidInputError(False, "not implement kwspaces for AutoObject")

    # @classproperty
    # def kwspaces(cls):
    #     """ OrderedDict representation of this search space.
    #     """
    #     return cls.__init__.kwspaces

    def sample(self):
        """Sample a configuration from this search space."""
        invalidInputError(False, "not implement sample for AutoObject")

    def __repr__(self):
        return 'AutoObject'


class List(NestedSpace):
    """
    Examples.

    ---------
        >>> sequence = space.List(
        >>>     space.space.Categorical('conv3x3', 'conv5x5', 'conv7x7'),
        >>>     space.space.Categorical('BatchNorm', 'InstanceNorm'),
        >>>     space.space.Categorical('relu', 'sigmoid'),
        >>> )
    """

    def __init__(self, *args, prefix=None):
        """
        Nested search space corresponding to an ordered list of hyperparameters.

        :param args: a list of search spaces.
            e.g.space.List(space.Int(1,2), space.Int(4,5),...)
        """
        self.data = [*args]
        super(List, self).__init__(prefix)

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    def __getstate__(self):
        return self.data

    def __setstate__(self, d):
        self.data = d

    def __getattribute__(self, s):
        try:
            x = super(List, self).__getattribute__(s)
        except AttributeError:
            pass
        else:
            return x
        x = self.data.__getattribute__(s)
        return x

    def sample(self, **config):
        """Sample a configuration from this search space."""
        ret = []
        kwspaces = self.kwspaces
        striped_keys = [k.split(SPLITTER)[0] for k in config.keys()]
        for idx, obj in enumerate(self.data):
            if isinstance(obj, NestedSpace):
                sub_config = _strip_config_space(config, prefix=str(idx))
                ret.append(obj.sample(**sub_config))
            elif isinstance(obj, SimpleSpace):
                ret.append(config[str(idx)])
            else:
                ret.append(obj)
        return ret

    @property
    def cs(self):
        """`ConfigSpace` representation of this search space."""
        cs = _new_cs(self.prefix)
        for k, v in enumerate(self.data):
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, str(k))
            elif isinstance(v, Space):
                hp = v.get_hp(name=str(k))
                _add_hp(cs, hp)
        return cs

    @property
    def kwspaces(self):
        """`OrderedDict` representation of this search space."""
        kw_spaces = OrderedDict()
        for idx, obj in enumerate(self.data):
            k = str(idx)
            if isinstance(obj, NestedSpace):
                kw_spaces[k] = obj
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}{}{}'.format(k, SPLITTER, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(obj, Space):
                kw_spaces[k] = obj
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


class Dict(NestedSpace):
    """
    Examples.

        >>> g = space.Dict(
        >>>         hyperparam1 = space.Categorical('alpha', 'beta'),
        >>>         hyperparam2 = space.Int(0, 3)
        >>>     )
        >>> print(g)
    """

    def __init__(self, **kwargs):
        """
        Nested search space for dictionary containing multiple hyperparameters.

        TODO: document how to set prefix for such params. may not be needed.
        :param kwargs: specify key and space values in form of keywork arguments.
            e.g.space.Dict(hp1=space.Int(1,2), hp2=space.Int(4,5))
        """
        self.data = EasyDict(kwargs)
        super(Dict, self).__init__(prefix=None)

    def __getattribute__(self, s):
        try:
            x = super(Dict, self).__getattribute__(s)
        except AttributeError:
            pass
        else:
            return x
        x = self.data.__getattribute__(s)
        return x

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, data):
        self.data[key] = data

    def __getstate__(self):
        return self.data

    def __setstate__(self, d):
        self.data = d

    @property
    def cs(self):
        """`ConfigSpace` representation of this search space."""
        cs = _new_cs(self.prefix)
        for k, v in self.data.items():
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, k)
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                _add_hp(cs, hp)
        return cs

    @property
    def kwspaces(self):
        """`OrderedDict` representation of this search space."""
        kw_spaces = OrderedDict()
        for k, obj in self.data.items():
            if isinstance(obj, NestedSpace):
                kw_spaces[k] = obj
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}{}{}'.format(k, SPLITTER, sub_k)
                    kw_spaces[new_k] = sub_v
                    kw_spaces[new_k] = sub_v
            elif isinstance(obj, Space):
                kw_spaces[k] = obj
        return kw_spaces

    def sample(self, **config):
        """Sample a configuration from this search space."""
        ret = {}
        ret.update(self.data)
        kwspaces = self.kwspaces
        kwspaces.update(config)
        striped_keys = [k.split(SPLITTER)[0] for k in config.keys()]
        for k, v in kwspaces.items():
            if k in striped_keys:
                if isinstance(v, NestedSpace):
                    sub_config = _strip_config_space(config, prefix=k)
                    ret[k] = v.sample(**sub_config)
                else:
                    ret[k] = v
        return ret

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


class Categorical(NestedSpace):
    """
    Examples.

        >>> a = space.Categorical('a', 'b', 'c', 'd')
        >>> b = space.Categorical('resnet50', AutoObj())
    """

    def __init__(self, *data, prefix=None):
        """
        Nested search space for hyperparameters which are categorical.

        Such a hyperparameter takes one value out of the discrete set
        of provided options. The first value in the list of options
        will be the default value that gets tried first during HPO.

        :param data: search space or python built-in objects.
            The first value will be the default value tried first during HPO.
            e.g.space.Dict(hp1=space.Int(1,2), hp2=space.Int(4,5))
        :param prefix: string (optional). This is useful for distinguishing
            the same hyperparameter in the same layer when a layer is
            used more than once in the model. Defaults to None.
        """
        self.data = [*data]
        super(Categorical, self).__init__(prefix)

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    @property
    def cs(self):
        """`ConfigSpace` representation of this search space."""
        cs = _new_cs(prefix=self.prefix)
        if len(self.data) == 0:
            return cs
        hp = create_categorical_hp(
            name='choice', choices=range(len(self.data)),
            meta={})
        _add_hp(cs, hp)
        for i, v in enumerate(self.data):
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, str(i))
        return cs

    def sample(self, **config):
        """Sample a configuration from this search space."""
        choice = config.pop('choice')
        if isinstance(self.data[choice], NestedSpace):
            # nested space: Categorical of AutoObjects
            min_config = _strip_config_space(config, prefix=str(choice))
            return self.data[choice].sample(**min_config)
        else:
            return self.data[choice]

    @property
    def kwspaces(self):
        """`OrderedDict` representation of this search space."""
        kw_spaces = OrderedDict()
        for idx, obj in enumerate(self.data):
            if isinstance(obj, NestedSpace):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}{}{}'.format(idx, SPLITTER, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


class Real(SimpleSpace):
    """
    Examples.

        >>> learning_rate = space.Real(0.01, 0.1, log=True)
    """

    def __init__(self, lower, upper, default=None, log=False, prefix=None):
        """
        Search space for numeric hyperparameter that takes continuous values.

        Example: space.Real(0.01, 0.1, log=True)

        :param lower: a float. The lower bound of the search space
            (minimum possible value of hyperparameter)
        :param upper: a float. The upper bound of the search space
            (maximum possible value of hyperparameter)
        :param default: a float (optional)
            Default value tried first during hyperparameter optimization
        :param log: boolean (optional). whether to search the values on
            a logarithmic rather than linear scale. This is useful for
            numeric hyperparameters (such as learning rates) whose
            search space spans many orders of magnitude.
        :param prefix: string (optional). This is useful for distinguishing
            the same hyperparameter in the same layer when a layer is
            used more than once in the model. Defaults to None.
        """
        self.lower = lower
        self.upper = upper
        self.log = log
        self._default = default
        super(Real, self).__init__(prefix)

    def get_hp(self, name):
        """Fetch particular hyperparameter based on its name."""
        return create_uniform_float_hp(
            name=name,
            lower=self.lower,
            upper=self.upper,
            default_value=self._default,
            log=self.log,
            meta={'prefix': self.prefix})


class Int(SimpleSpace):
    """
    Examples.

        >>> range = space.Int(0, 100)
    """

    def __init__(self, lower, upper, default=None, prefix=None):
        """
        Search space for numeric hyperparameter that takes integer values.

        :param lower: int. The lower bound of the search space
            (minimum possible value of hyperparameter)
        :param upper: int. The upper bound of the search space
            (maximum possible value of hyperparameter)
        :param default: int (optional) Default value tried first time
            during hyperparameter optimization
        :param prefix: string (optional). This is useful for distinguishing
            the same hyperparameter in the same layer when a layer is
            used more than once in the model. Defaults to None.
        """
        self.lower = lower
        self.upper = upper
        self._default = default
        super(Int, self).__init__(prefix)

    def get_hp(self, name):
        """Fetch particular hyperparameter based on its name."""
        return create_uniform_int_hp(
            name=name,
            lower=self.lower,
            upper=self.upper,
            default_value=self._default,
            meta={'prefix': self.prefix})


class Bool(Int):
    """
    Examples.

        >>> pretrained = space.Bool()
    """

    def __init__(self, default=None, prefix=None):
        """
        Search space for hyperparameter that is either True or False.

        `space.Bool()` serves as shorthand for: `space.Categorical(True, False)`
        """
        super(Bool, self).__init__(0, 1, default=default, prefix=prefix)


class SingleParam(object):
    """A object to hold single params spaces which does not belong to\
    any AutoObject."""

    delimiter = u'.'

    def __init__(self, argname, param):
        self.argname = argname
        self.param = param
        self.cs = _new_cs()
        if isinstance(self.param, SimpleSpace):
            _add_hp(self.cs, param.get_hp(argname))
        elif isinstance(self.param, NestedSpace):
            _add_cs(self.cs, param.cs, self.argname, delimiter='.', parent_hp=None)
        else:
            # usually should not fall to this path
            self.cs = param.cs

    def sample(self, **config):
        if isinstance(self.param, SimpleSpace):
            new_params = config.get(self.argname, None)
        elif isinstance(self.param, NestedSpace):
            sub_config = _strip_config_space(config, prefix=self.argname)
            new_params = self.param.sample(**sub_config)
        return new_params


def _strip_config_space(config, prefix):
    # filter out the config with the corresponding prefix
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix) + 1:]] = v
    return new_config


def _new_cs(prefix=None):
    return create_configuration_space(meta={'prefix': prefix})


def _get_cs_prefix(cs):
    return cs.meta.setdefault('prefix', None)


def _set_cs_prefix(cs, prefix):
    cs.meta['prefix'] = prefix


def _get_hp_prefix(hp):
    return hp.meta.setdefault('prefix', None)


def _set_hp_prefix(hp, prefix):
    hp.meta['prefix'] = prefix


def _add_hp(cs, hp):
    if hp.name in cs._hyperparameters:
        cs._hyperparameters[hp.name] = hp
    else:
        cs.add_hyperparameter(hp)


def _add_cs(master_cs, sub_cs, prefix, delimiter='.', parent_hp=None):
    """Add the params from sub_cs to master_cs."""
    new_parameters = []
    for hp in sub_cs.get_hyperparameters():
        new_parameter = copy.deepcopy(hp)
        # Allow for an empty top-level parameter
        if new_parameter.name == '':
            new_parameter.name = prefix
        elif not prefix == '':
            new_parameter.name = "{}{}{}".format(
                prefix, SPLITTER, new_parameter.name)
            # further add the sub_cs prefix onto the param
            sub_cs_prefix = _get_cs_prefix(sub_cs)
            if not sub_cs_prefix == '':
                _set_hp_prefix(new_parameter, sub_cs_prefix)
        new_parameters.append(new_parameter)
    for hp in new_parameters:
        _add_hp(master_cs, hp)


def _rm_hp(cs, k):
    if k in cs._hyperparameters:
        cs._hyperparameters.pop(k)
    for hp in cs.get_hyperparameters():
        if hp.name.startswith('{}'.format(k)):
            cs._hyperparameters.pop(hp.name)
