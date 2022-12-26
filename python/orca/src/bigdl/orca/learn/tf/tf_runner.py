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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import closing
import json
import logging
import os
import socket

import ray

from bigdl.dllib.utils.log4Error import invalidInputError

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Callable
    from tensorflow.compat.v1.estimator import TrainSpec, EvalSpec

logger = logging.getLogger(__name__)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _try_import_strategy():
    """Late import for Tesnorflow"""
    import tensorflow as tf
    return tf.distribute.experimental.MultiWorkerMirroredStrategy


class TFRunner:
    """Manages a TensorFlow estimator for training."""

    def __init__(
        self,
        model_fn: Callable,
        model_dir: Optional[str]=None,
        config: Optional[Dict[str, Any]]=None,
        params: Optional[Dict[str, Any]]=None,
        warm_start_from: Optional[str]=None
    ) -> None:
        """Initializes the runner.
        Args:
            model_creator (dict -> Model): see tf_trainer.py.
            data_creator (dict -> tf.Dataset, tf.Dataset): see tf_trainer.py.
            config (dict): see tf_trainer.py.
            verbose (bool): Outputs training data if true.
        """
        import tensorflow.compat.v1 as tf
        tf.logging.set_verbosity(tf.logging.INFO)

        self.model_fn = model_fn
        tf.gfile.MakeDirs(model_dir)
        self.model_dir = model_dir
        self.config = {} if config is None else config
        self.inter_op_parallelism = self.config.pop("inter_op_parallelism", 1)
        self.intra_op_parallelism = self.config.pop("intra_op_parallelism", 1)
        self.params = params
        self.warm_start_from = warm_start_from

    def setup(self) -> None:
        import tensorflow.compat.v1 as tf
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism)
        os.environ["KMP_BLOCKING_TIME"] = self.config.get("KMP_BLOCKING_TIME",
                                                          os.environ.get("KMP_BLOCKING_TIME", "0"))

    def setup_local(self) -> None:
        """Initializes the model."""
        self.backend = "tf-local"
        self.size = 1
        self.rank = 0
        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        self.strategy = ds_context.get_strategy()

    def setup_distributed(
        self,
        urls: List[str],
        world_rank: int,
        world_size: int
    ) -> None:
        """Sets up TensorFLow distributed environment and initializes the model.
        Args:
            urls (str): the URLs that each node uses to connect.
            world_rank (int): the index of the runner.
            world_size (int): the total number of runners.
        """
        import tensorflow.compat.v1 as tf
        invalidInputError(len(urls) == world_size, "expect len(urls) == world_size")
        tf_config = {
            "cluster": {
                "worker": urls
            },
            "task": {
                "index": world_rank,
                "type": "worker"
            }
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        no_proxy = os.environ.get("no_proxy", "")
        ips = [url.split(":")[0] for url in urls]
        os.environ["no_proxy"] = ",".join(ips) + "," + no_proxy

        MultiWorkerMirroredStrategy = _try_import_strategy()

        # MultiWorkerMirroredStrategy handles everything for us, from
        # sharding the dataset (or even sharding the data itself if the loader
        # reads files from disk) to merging the metrics and weight updates
        #
        # worker 0 is the "chief" worker and will handle the map-reduce
        # every worker ends up with the exact same metrics and model
        # after model.fit
        #
        # because of this, we only really ever need to query its state
        self.strategy = MultiWorkerMirroredStrategy()

        logger.debug("Creating model with MultiWorkerMirroredStrategy")

        # todo check keys in self.config
        dist_config = tf.estimator.RunConfig(
            train_distribute=self.strategy,
            model_dir=self.model_dir,
            **self.config)

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=dist_config,
            model_dir=self.model_dir,
            params=self.params,
            warm_start_from=self.warm_start_from)

        # For use in model.evaluate()
        self.local_model = None
        self.backend = "tf-distributed"
        self.size = world_size
        self.rank = world_rank

    def train_and_evaluate(self,
                           train_spec: "TrainSpec",
                           eval_spec: "EvalSpec") -> Any:
        import tensorflow.compat.v1 as tf
        result = tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        return result

    def shutdown(self) -> None:
        """Attempts to shut down the worker."""
        del self.estimator

    def get_node_ip(self) -> str:
        """Returns the IP address of the current node."""
        return ray._private.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()
