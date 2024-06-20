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
import itertools
import logging
import pickle

import numpy as np
import ray
from bigdl.dllib.utils.file_utils import enable_multi_fs_load, enable_multi_fs_save

from bigdl.orca.data.ray_xshards import RayXShards
from bigdl.orca.learn.dl_cluster import RayDLCluster
from bigdl.orca.learn.tf2.tf_runner import TFRunner
from bigdl.orca.learn.ray_estimator import Estimator as OrcaRayEstimator
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe, make_data_creator
from bigdl.orca.data.utils import process_spark_xshards
from bigdl.orca.ray import RayContext

logger = logging.getLogger(__name__)


class TensorFlow2Estimator(OrcaRayEstimator):
    def __init__(self,
                 model_creator,
                 compile_args_creator=None,
                 config=None,
                 verbose=False,
                 backend="tf2",
                 workers_per_node=1,
                 cpu_binding=False):
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose

        ray_ctx = RayContext.get()
        if "batch_size" in self.config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate function of the estimator instead.")

        if "inter_op_parallelism" not in self.config:
            self.config["inter_op_parallelism"] = 1

        if "intra_op_parallelism" not in self.config:
            self.config["intra_op_parallelism"] = ray_ctx.ray_node_cpu_cores // workers_per_node

        if backend == "horovod":
            assert compile_args_creator is not None, "compile_args_creator should not be None," \
                                                     " when backend is set to horovod"

        params = {
            "model_creator": model_creator,
            "compile_args_creator": compile_args_creator,
            "config": self.config,
            "verbose": self.verbose,
        }

        if backend == "tf2":
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node

            self.cluster = RayDLCluster(
                num_workers=num_nodes,
                worker_cores=cores_per_node,
                worker_cls=TFRunner,
                worker_param=params,
                cpu_binding=cpu_binding
            )
            self.remote_workers = self.cluster.get_workers()
            ips = ray.get(
                [worker.get_node_ip.remote() for worker in self.remote_workers])
            ports = ray.get(
                [worker.find_free_port.remote() for worker in self.remote_workers])

            urls = ["{ip}:{port}".format(ip=ips[i], port=ports[i])
                    for i in range(len(self.remote_workers))]
            ray.get([worker.setup.remote() for worker in self.remote_workers])
            # Get setup tasks in order to throw errors on failure
            ray.get([
                worker.setup_distributed.remote(urls, i, len(self.remote_workers))
                for i, worker in enumerate(self.remote_workers)])
        elif backend == "horovod":
            # it is necessary to call self.run first to set horovod environment
            from bigdl.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            horovod_runner = HorovodRayRunner(ray_ctx,
                                              worker_cls=TFRunner,
                                              worker_param=params,
                                              workers_per_node=workers_per_node)
            horovod_runner.run(lambda: print("worker initialized"))
            self.remote_workers = horovod_runner.remote_workers
            ray.get([worker.setup.remote() for worker in self.remote_workers])
            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)])
        else:
            raise Exception("Only \"tf2\" and \"horovod\" are legal "
                            "values of backend, but got {}".format(backend))

        self.num_workers = len(self.remote_workers)

    def fit(self, data, epochs=1, batch_size=32, verbose=1,
            callbacks=None, validation_data=None, class_weight=None,
            steps_per_epoch=None, validation_steps=None, validation_freq=1,
            data_config=None, feature_cols=None,
            label_cols=None, tops=None):
        """
        Train this tensorflow model with train data.

        :param data: train data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param epochs: Number of epochs to train the model. Default: 1.
        :param batch_size: Batch size used for training. Default: 32.
        :param verbose: Prints output of one model if true.
        :param callbacks: List of Keras compatible callbacks to apply during training.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param class_weight: Optional dictionary mapping class indices (integers) to a weight
               (float) value, used for weighting the loss function. This can be useful to tell
               the model to "pay more attention" to samples from an under-represented class.
        :param steps_per_epoch: Total number of steps (batches of samples) before declaring one
               epoch finished and starting the next epoch. If `steps_pre_epoch` is `None`, the
               epoch will run until the input dataset is exhausted. When passing an infinitely
               repeating dataset, you must specify the `step_per_epoch` argument.
        :param validation_steps: Total number of steps (batches of samples) to draw before stopping
               when performing validation at the end of every epoch. Default: None.
        :param validation_freq: Only relevant if validation data is provided. Integer of
               `collections_abc.Container` instance (e.g. list, tuple, etc.). If an integer,
               specifies how many training epochs to run before a new validation run is performed,
               e.g. `validation_freq=2` runs validation every 2 epochs. If a Container, specifies
               the epochs on which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
               validation at the end of the 1st, 2nd, and 10th epochs.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is a Spark DataFrame or
               an XShards of Pandas DataFrame.
               Default: None.
        :return:
        """
        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            data_config=data_config,
            tops=tops
        )

        from bigdl.orca.data import SparkXShards
        data, validation_data = maybe_dataframe_to_xshards(data, validation_data,
                                                           feature_cols, label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           accept_str_col=True)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")
            ray_xshards = process_spark_xshards(data, self.num_workers)

            if validation_data is None:
                def transform_func(worker, partition_refs):
                    params["data_creator"] = make_data_creator(partition_refs)
                    return worker.step.remote(**params)

                worker_stats = ray_xshards.reduce_partitions_for_actors(self.remote_workers,
                                                                        transform_func)
            else:
                val_ray_xshards = process_spark_xshards(validation_data, self.num_workers)

                def zip_func(worker, this_partition_refs, that_partition_refs):
                    params["data_creator"] = make_data_creator(this_partition_refs)
                    params["validation_data_creator"] = \
                        make_data_creator(that_partition_refs)
                    return worker.step.remote(**params)

                worker_stats = ray_xshards.zip_reduce_shards_with_actors(val_ray_xshards,
                                                                         self.remote_workers,
                                                                         zip_func)
        else:
            params["data_creator"] = data
            params["validation_data_creator"] = validation_data
            params_list = [params] * self.num_workers

            worker_stats = ray.get([self.remote_workers[i].step.remote(**params_list[i])
                                    for i in range(self.num_workers)])
            worker_stats = list(itertools.chain.from_iterable(worker_stats))
        stats = worker_stats[0].copy()
        return stats

    def evaluate(self, data, batch_size=32, num_steps=None, verbose=1,
                 sample_weight=None, callbacks=None, data_config=None,
                 feature_cols=None, label_cols=None):
        """
        Evaluates the model on the validation data set.

        :param data: evaluate data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param batch_size: Batch size used for evaluation. Default: 32.
        :param num_steps: Total number of steps (batches of samples) before declaring the evaluation
               round finished. Ignored with the default value of `None`.
        :param verbose: Prints output of one model if true.
        :param sample_weight: Optional Numpy array of weights for the training samples, used for
               weighting the loss function. You can either pass a flat (1D) Numpy array with the
               same length as the input samples (1:1 mapping between weights and samples), or in
               the case of temporal data, you can pass a 2D array with shape (samples,
               sequence_length), to apply a different weight to every timestep of every sample.
        :param callbacks: List of Keras compatible callbacks to apply during evaluation.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is a Spark DataFrame or
               an XShards of Pandas DataFrame.
               Default: None.
        :return: validation result
        """
        logger.info("Starting validation step.")
        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=num_steps,
            callbacks=callbacks,
            data_config=data_config,
        )
        from bigdl.orca.data import SparkXShards

        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             accept_str_col=True)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

            data = data
            if data.num_partitions() != self.num_workers:
                data = data.repartition(self.num_workers)

            ray_xshards = RayXShards.from_spark_xshards(data)

            def transform_func(worker, partition_refs):
                params["data_creator"] = make_data_creator(partition_refs)
                return worker.validate.remote(**params)

            worker_stats = ray_xshards.reduce_partitions_for_actors(self.remote_workers,
                                                                    transform_func)
        else:  # data_creator functions; should return Iter or DataLoader
            params["data_creator"] = data
            params_list = [params] * self.num_workers

            worker_stats = ray.get([w.validate.remote(**params_list[i])
                                    for i, w in enumerate(self.remote_workers)])
            worker_stats = list(itertools.chain.from_iterable(worker_stats))
        stats = worker_stats[0].copy()
        return stats

    def _predict_spark_xshards(self, xshards, params):
        ray_xshards = RayXShards.from_spark_xshards(xshards)

        def transform_func(worker, shards_ref):
            params["data_creator"] = make_data_creator(shards_ref)
            return worker.predict.remote(**params)

        pred_shards = ray_xshards.transform_shards_with_actors(self.remote_workers,
                                                               transform_func)
        spark_xshards = pred_shards.to_spark_xshards()
        return spark_xshards

    def predict(self, data, batch_size=None, verbose=1,
                steps=None, callbacks=None, data_config=None,
                feature_cols=None):
        """
        Predict the input data

        :param data: predict input data.  It can be XShards or Spark DataFrame.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
        :param batch_size: Batch size used for inference. Default: None.
        :param verbose: Prints output of one model if true.
        :param steps: Total number of steps (batches of samples) before declaring the prediction
               round finished. Ignored with the default value of None.
        :param callbacks: List of Keras compatible callbacks to apply during prediction.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :return:
        """
        logger.info("Starting predict step.")
        params = dict(
            verbose=verbose,
            batch_size=batch_size,
            steps=steps,
            callbacks=callbacks,
            data_config=data_config,
        )
        from bigdl.orca.data import SparkXShards
        from pyspark.sql import DataFrame

        if isinstance(data, DataFrame):
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict",
                                              accept_str_col=True)
            pred_shards = self._predict_spark_xshards(xshards, params)
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        elif isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols)
            pred_shards = self._predict_spark_xshards(data, params)
            result = update_predict_xshards(data, pred_shards)
        else:
            raise ValueError("Only xshards or Spark DataFrame is supported for predict")

        return result

    def get_model(self):
        """
        Returns the learned model.

        :return: the learned model.
        """
        state_refs = [w.get_state.remote() for w in self.remote_workers]
        state = ray.get(state_refs[0])
        return self._get_model_from_state(state)

    @enable_multi_fs_save
    def save(self, checkpoint):
        """
        Saves the model at the provided checkpoint.

        :param checkpoint: (str) Path to the target checkpoint file.
        """

        # Some model might need to aggregate variables during checkpointing
        # which requires both the chief and workers to participate in the
        # allreduce communication protocol.
        # So we need to call get_state on every remote workers, otherwise
        # it might get stuck
        state_refs = [w.get_state.remote() for w in self.remote_workers]

        state = ray.get(state_refs[0])

        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)

        return checkpoint

    @enable_multi_fs_load
    def load(self, checkpoint, **kwargs):
        """
        Loads the model from the provided checkpoint.

        :param checkpoint: (str) Path to target checkpoint file.

        """
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)

        state_id = ray.put(state)
        ray.get([worker.set_state.remote(state_id) for worker in self.remote_workers])

    def shutdown(self):
        """
        Shuts down workers and releases resources.
        """
        for worker in self.remote_workers:
            worker.shutdown.remote()
            worker.__ray_terminate__.remote()

    def _get_model_from_state(self, state):
        """Creates model and load weights from state"""

        # keep the same behavior as `set_state` in `load` do
        model = self.model_creator(self.config)
        model.set_weights(state["weights"])

        return model
