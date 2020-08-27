#
# Copyright 2018 Analytics Zoo Authors.
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

import ray
from ray.util.sgd.torch import TorchTrainer
from zoo.ray import RayContext


class PyTorchTrainer(object):
    def __init__(self, model_creator, data_creator, optimizer_creator,
                 loss_creator=None, scheduler_creator=None, training_operator_cls=None,
                 initialization_hook=None, config=None, num_workers=None,
                 use_fp16=False, use_tqdm=False, scheduler_step_freq="batch"):
        # Lift TorchTrainer to an Actor so that its local worker would be
        # created on the cluster as well.
        RemoteTrainer = ray.remote(TorchTrainer)

        # check whether there has been an active RayContext and get it.
        ray_ctx = RayContext.get()
        if not num_workers:
            num_workers = ray_ctx.num_ray_nodes
        self.trainer = RemoteTrainer.remote(model_creator=model_creator,
                                            data_creator=data_creator,
                                            optimizer_creator=optimizer_creator,
                                            loss_creator=loss_creator,
                                            scheduler_creator=scheduler_creator,
                                            training_operator_cls=training_operator_cls,
                                            initialization_hook=initialization_hook,
                                            config=config,
                                            num_workers=num_workers,
                                            backend="gloo",
                                            use_fp16=use_fp16,
                                            use_tqdm=use_tqdm,
                                            scheduler_step_freq=scheduler_step_freq)

    def train(self, nb_epoch=1):
        """Trains a PyTorch model for several epochs."""
        stats_list = list()
        for i in range(nb_epoch):
            stats = ray.get(self.trainer.train.remote())
            stats_list.append(stats)
        return stats_list

    def validate(self, num_steps=None, profile=False, reduce_results=True, info=None):
        return ray.get(self.trainer.validate.remote(num_steps, profile, reduce_results, info))

    def get_model(self):
        """Returns the learned model(s)."""
        return ray.get(self.trainer.get_model.remote())

    def shutdown(self, force=False):
        ray.get(self.trainer.shutdown.remote(force))
