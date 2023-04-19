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


import cloudpickle
import tensorflow as tf
import numpy as np
from functools import wraps, partial
from tempfile import TemporaryDirectory
import os
import json

from bigdl.nano.utils.common import schedule_processors
from bigdl.nano.tf.keras.distributed_utils import _find_free_port
from tensorflow.keras.losses import Loss
import warnings
from bigdl.nano.utils.common import invalidInputError


def nano_bf16(func):
    """A decorator to realize mixed precision on customized training loop."""
    # todo check the func signature
    @wraps(func)
    def wrapper(*inner_args):
        new_args = []
        for arg in inner_args:
            if isinstance(arg, (tf.Tensor, np.ndarray)):
                arg = tf.cast(arg, tf.bfloat16)
            new_args.append(arg)
        return func(*new_args)
    return wrapper


class nano_multiprocessing(object):
    """A decorator to realize nano_multiprocessing training on customized training step."""

    def __init__(self, func):
        """Initialize the training step function."""
        self.func = func

    def __call__(self, *args, mirrored_strategy=None, **kwargs):
        """Run distribution strategy for multi-process training."""
        # TODO: to validate if we really could support kwargs
        per_replica_losses = mirrored_strategy.run(self.func, args=args, kwargs=kwargs)
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                        axis=None)


def nano(num_processes):
    """
    A decorator to run customized training loop on multiple processes.

    :param num_processes: int, number of processes.
    """
    def decorator(func):
        return _Nano_Customized_Training(func, num_processes)
    return decorator


class _Nano_Customized_Training(object):
    def __init__(self, func, nproc):
        self.func = func
        self.nproc = nproc

    def __call__(self, *args, **kwargs):
        new_args = []
        from bigdl.nano.utils.tf import MultiprocessingBackend
        backend = MultiprocessingBackend()

        main_model = None

        with TemporaryDirectory() as temp_dir:
            for i, arg in enumerate(args):
                # save the model
                if isinstance(arg, tf.Module):
                    arg.save(os.path.join(temp_dir, f'args_{i}'))
                    new_args.append(("model", os.path.join(temp_dir, f'args_{i}')))
                    main_model = arg
                    continue

                # save the optimizer
                if isinstance(arg, tf.keras.optimizers.Optimizer):
                    with open(os.path.join(temp_dir, f"args_{i}.pkl"), 'wb') as f:
                        cloudpickle.dump(arg, f)
                    new_args.append(("optimizer", os.path.join(temp_dir, f'args_{i}.pkl')))
                    continue

                # save the loss
                if isinstance(arg, tf.keras.losses.Loss):
                    with open(os.path.join(temp_dir, f"args_{i}.pkl"), 'wb') as f:
                        cloudpickle.dump(arg, f)
                    new_args.append(("loss", os.path.join(temp_dir, f'args_{i}.pkl')))
                    continue

                # serialize the dataset
                if isinstance(arg, tf.data.Dataset):
                    warnings.warn('If dataset is created by `from_generator`, please initiate '
                                  '`_GeneratorState` of the dataset first, that is '
                                  'dataset._GeneratorState = dataset._GeneratorState(generator). '
                                  'Otherwise, there exists errors because of a known limitation: '
                                  'https://www.tensorflow.org/api_docs/python/tf/data/Dataset'
                                  '#from_generator')
                    if hasattr(arg, '_GeneratorState') and \
                       hasattr(arg._GeneratorState, '_generator'):
                        # support dataset created by `from_generator`
                        train_ds_gen = arg._GeneratorState._generator
                        train_ds_signature = arg.element_spec
                        new_args.append(("dataset", train_ds_gen, train_ds_signature))
                        continue
                    else:
                        from tensorflow.python.distribute.coordinator.values import \
                            serialize_dataset_to_graph

                        train_ds_def = serialize_dataset_to_graph(arg).numpy()
                        train_elem_spec = arg.element_spec
                        new_args.append(("dataset", train_ds_def, train_elem_spec))
                        continue

                with open(os.path.join(temp_dir, f"args_{i}.pkl"), 'wb') as f:
                    cloudpickle.dump(arg, f)
                    new_args.append(("others", os.path.join(temp_dir, f"args_{i}.pkl")))

            # check necessary objects
            arg_instance = list(map(lambda x: x[0], new_args))
            invalidInputError("model" in arg_instance, "Please check if a model inherited from "
                                                       "tensorflow.keras.Model or "
                                                       "bigdl.nano.tf.keras.Model is inputted as "
                                                       "parameter in train function.")
            invalidInputError("optimizer" in arg_instance, "Please check if optimizer is inputted"
                                                           " as parameter in train function.")
            invalidInputError("loss" in arg_instance, "Please check if loss is inputted as "
                                                      "parameter in train function. Moreover, if "
                                                      "you use a customized loss, please add "
                                                      "'nano_multiprocessing_loss' decorator.")
            invalidInputError("dataset" in arg_instance, "Please check if tensorflow.data.Dataset "
                                                         "is inputted as parameter in train "
                                                         "function. Moreover, if dataset is "
                                                         "created by 'from_generator', please init"
                                                         " '_GeneratorState' of dataset first.")

            target_path = os.path.join(temp_dir, "target.pkl")
            with open(target_path, 'wb') as f:
                cloudpickle.dump(self.func, f)

            ports = set()
            while len(ports) < self.nproc:
                ports.add(_find_free_port())
            ports = list(ports)
            worker_list = [f"localhost:{p}" for p in ports]

            # TODO: this env mainly for core affinity and allocation limit
            # while does not work for stock TF
            envs = schedule_processors(self.nproc)

            for i, env in enumerate(envs):
                env.update({
                    "TF_CONFIG": json.dumps(
                        {
                            'cluster': {
                                'worker': worker_list
                            },
                            'task': {'type': 'worker', 'index': i}
                        }),
                    'no_proxy': "localhost"
                })

            histrories = backend.run(target=_train_func,
                                     args=(target_path, *new_args),
                                     nprocs=self.nproc,
                                     envs=envs)

            main_model.load_weights('trained_model_weights')

        return histrories[0]


def _train_func(target_path, *args):
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    actrual_args = [None] * len(args)
    new_model = None

    with mirrored_strategy.scope():
        for i, arg in enumerate(args):
            if arg[0] == "model":
                actrual_args[i] = tf.keras.models.load_model(arg[1])
                new_model = actrual_args[i]
                continue
            # deserialize optimizer
            if arg[0] == "optimizer":
                with open(arg[1], 'rb') as f:
                    actrual_args[i] = cloudpickle.load(f)
                continue
            # deserialize dataset
            if arg[0] == "dataset":
                try:
                    from tensorflow.python.distribute.coordinator.values import \
                        deserialize_dataset_from_graph
                    original_dataset = deserialize_dataset_from_graph(arg[1], arg[2])
                except ValueError:
                    original_dataset = tf.data.Dataset.from_generator(arg[1],
                                                                      output_signature=arg[2])
                actrual_args[i] = mirrored_strategy.experimental_distribute_dataset(
                    original_dataset)
                continue

            if arg[0] == "loss":
                with open(arg[1], 'rb') as f:
                    original_loss_object = cloudpickle.load(f)
                    original_loss_object.reduction = tf.keras.losses.Reduction.NONE

                def loss_object(*args, **kwargs):
                    per_example_loss = original_loss_object(*args, **kwargs)
                    if per_example_loss.shape == [] or per_example_loss.shape[0] == 0:
                        size = mirrored_strategy.num_replicas_in_sync
                        return tf.math.reduce_sum(per_example_loss) / size
                    else:
                        size = per_example_loss.shape[0] * mirrored_strategy.num_replicas_in_sync
                        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=size)
                actrual_args[i] = loss_object
                continue
            # deserialize others
            if arg[0] == "others":
                with open(arg[1], 'rb') as f:
                    actrual_args[i] = cloudpickle.load(f)
                    if callable(actrual_args[i]) and \
                       isinstance(actrual_args[i], nano_multiprocessing):
                        actrual_args[i] = partial(actrual_args[i],
                                                  mirrored_strategy=mirrored_strategy)

        with open(target_path, 'rb') as f:
            target_func = cloudpickle.load(f)

        try:
            res = target_func(*actrual_args)
        except TypeError:
            invalidInputError(False, "Please check if you have added 'nano_multiprocessing' "
                                     "decorator to the train_step function.")

        task_id = mirrored_strategy.cluster_resolver.task_id
        if task_id == 0:
            path = os.path.join('trained_model_weights')
            new_model.save_weights(path, overwrite=True)

    try:
        # Related to https://github.com/tensorflow/tensorflow/issues/50487, to avoid such error
        import atexit
        atexit.register(mirrored_strategy._extended._cross_device_ops._pool.close)
        atexit.register(mirrored_strategy._extended._host_cross_device_ops._pool.close)
    except AttributeError:
        pass
    return res


def nano_multiprocessing_loss(**kwargs):
    """A decorator to run customized loss on multiple processes."""
    def decorator(func):
        return Nano_Customized_Loss(func, **kwargs)
    return decorator


class Nano_Customized_Loss(Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(self, func, reduction=None, name=None, **kwargs):
        """Initialize the loss function."""
        if reduction is None:
            reduction = tf.keras.losses.Reduction.NONE
        super().__init__(reduction=reduction, name=name)
        self.fn = func
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Run the loss function for multi-process training."""
        return self.fn(y_true, y_pred, **self._fn_kwargs)
