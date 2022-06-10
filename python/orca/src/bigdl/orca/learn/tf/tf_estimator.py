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

from bigdl.orca.ray import RayContext
from bigdl.orca.learn.tf.tf_runner import TFRunner
import ray
from bigdl.orca.learn.dl_cluster import RayDLCluster


class TFEstimator():
    def __init__(self,
                 model_fn,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,
                 workers_per_node=1,
                 cpu_binding=False,
                 ):
        """
        :param model_fn: Model function. Follows the signature:

            * Args:

                * `features`: This is the first item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same.
                * `labels`: This is the second item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same (for multi-head models).
                    If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                    be passed. If the `model_fn`'s signature does not accept
                    `mode`, the `model_fn` must still be able to handle
                    `labels=None`.
                * `mode`: Optional. Specifies if this training, evaluation or
                    prediction. See `tf.estimator.ModeKeys`.
                * `params`: Optional `dict` of hyperparameters.  Will receive what
                    is passed to Estimator in `params` parameter. This allows
                    to configure Estimators from hyper parameter tuning.
                * `config`: Optional `estimator.RunConfig` configuration object.

            * Returns:
                `tf.estimator.EstimatorSpec`

        :param model_dir: Directory to save model parameters, graph and etc. This can
            also be used to load checkpoints from the directory into an estimator to
            continue training a previously saved model. If `PathLike` object, the
            path will be resolved. If `None`, the model_dir in `config` will be used
            if set. If both are set, they must be same. If both are `None`, a
            temporary directory will be used.
        :param config: Optional. Params dictionary for `estimator.RunConfig`.
            E.g. {"keep_checkpoint_max":5, "save_checkpoints_steps":1000}, as well as other session
             configs including "inter_op_parallelism" and "intra_op_parallelism".
        :param params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
        :param warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                       warm-start from, or a `tf.estimator.WarmStartSettings`
                       object to fully configure warm-starting.  If the string
                       filepath is provided instead of a
                       `tf.estimator.WarmStartSettings`, then all variables are
                       warm-started, and it is assumed that vocabularies
                       and `tf.Tensor` names are unchanged.
        :param workers_per_node: (Int) worker number on each node. default: 1.
        :param cpu_binding: (bool) Whether to binds threads to specific CPUs. Default: False

        """
        self.config = {} if config is None else config

        ray_ctx = RayContext.get()
        if "batch_size" in self.config:
            from bigdl.dllib.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "Please do not specify batch_size in config. Input batch_size in the"
                              " fit/evaluate function of the estimator instead.")

        if "inter_op_parallelism" not in self.config:
            self.config["inter_op_parallelism"] = 1

        if "intra_op_parallelism" not in self.config:
            self.config["intra_op_parallelism"] = ray_ctx.ray_node_cpu_cores // workers_per_node

        params = dict(
            model_fn=model_fn,
            model_dir=model_dir,
            config=self.config,
            params=params,
            warm_start_from=warm_start_from
        )

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
        self.num_workers = len(self.remote_workers)

    def train_and_evaluate(self,
                           train_spec,
                           eval_spec):
        """
        Train and evaluate the estimator.

        :param train_spec: A TrainSpec instance to specify the training specification.
        :param eval_spec: A EvalSpec instance to specify the evaluation and export specification.

        Returns:
            A tuple of the result of the evaluate call to the Estimator and the export results using
             the specified ExportStrategy. Currently, the return value is undefined for distributed
            training mode.
        """
        params = dict(
            train_spec=train_spec,
            eval_spec=eval_spec,
        )
        results = ray.get([worker.train_and_evaluate.remote(**params)
                          for worker in self.remote_workers])
        return results
