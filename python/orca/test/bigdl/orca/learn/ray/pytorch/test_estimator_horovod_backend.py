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
from zoo.examples.orca.learn.horovod.pytorch_estimator import (model_creator,
                                                               optimizer_creator,
                                                               scheduler_creator,
                                                               train_data_creator,
                                                               validation_data_creator)
from zoo.orca.learn.pytorch import Estimator
from unittest import TestCase
import pytest
import torch
import torch.nn as nn
import os
from tempfile import TemporaryDirectory


class TestPyTorchEstimator(TestCase):
    def test_train(self):
        estimator = Estimator.from_torch(
            model=model_creator,
            optimizer=optimizer_creator,
            loss=nn.MSELoss(),
            scheduler_creator=scheduler_creator,
            config={
                "lr": 1e-2,  # used in optimizer_creator
                "hidden_size": 1  # used in model_creator
            }, backend="horovod", workers_per_node=2)
        stats1 = estimator.fit(train_data_creator, batch_size=4, epochs=5)
        train_loss1 = stats1[-1]["train_loss"]
        validation_loss1 = estimator.evaluate(validation_data_creator)["val_loss"]

        stats2 = estimator.fit(train_data_creator, batch_size=4, epochs=3)
        train_loss2 = stats2[-1]["train_loss"]
        validation_loss2 = estimator.evaluate(validation_data_creator)["val_loss"]

        # Verify syncing weights, i.e. the two workers have the same weights after training
        import ray
        import numpy as np
        remote_workers = estimator.estimator.remote_workers
        state_dicts = ray.get([worker.state_dict.remote() for worker in remote_workers])
        weights = [state["models"] for state in state_dicts]
        worker1_weights = weights[0][0]
        worker2_weights = weights[1][0]
        for layer in list(worker1_weights.keys()):
            assert np.allclose(worker1_weights[layer].numpy(),
                               worker2_weights[layer].numpy())

        assert train_loss2 <= train_loss1, (train_loss2, train_loss1)
        # todo this test maybe too strict, need to further check
        # assert validation_loss2 <= validation_loss1, (validation_loss2,
        #                                               validation_loss1)
        estimator.shutdown()

    def test_horovod_initialized_correctly(self):
        estimator = Estimator.from_torch(
            model=model_creator,
            optimizer=optimizer_creator,
            loss=nn.MSELoss(),
            scheduler_creator=scheduler_creator,
            config={
                "lr": 1e-2,  # used in optimizer_creator
                "hidden_size": 1  # used in model_creator
            }, backend="horovod", workers_per_node=2)

        def get_size():
            import horovod.torch as hvd
            return hvd.size()
        results = estimator.estimator.horovod_runner.run(get_size)
        assert results == [2, 2]

        def get_rank():
            import horovod.torch as hvd
            return hvd.rank()
        results = estimator.estimator.horovod_runner.run(get_rank)
        results = sorted(results)
        assert results == [0, 1]
        estimator.shutdown()

    def test_save_and_restore(self):
        estimator1 = Estimator.from_torch(
            model=model_creator,
            optimizer=optimizer_creator,
            loss=nn.MSELoss(),
            scheduler_creator=scheduler_creator,
            config={
                "lr": 1e-2,  # used in optimizer_creator
                "hidden_size": 1  # used in model_creator
            }, backend="horovod")
        with TemporaryDirectory() as tmp_path:
            estimator1.fit(train_data_creator, batch_size=4, epochs=1)
            checkpoint_path = os.path.join(tmp_path, "checkpoint")
            estimator1.save(checkpoint_path)

            model1 = estimator1.get_model()

            estimator1.shutdown()

            estimator2 = Estimator.from_torch(
                model=model_creator,
                optimizer=optimizer_creator,
                loss=nn.MSELoss(),
                scheduler_creator=scheduler_creator,
                config={
                    "lr": 1e-2,  # used in optimizer_creator
                    "hidden_size": 1  # used in model_creator
                }, backend="horovod")
            estimator2.load(checkpoint_path)

            model2 = estimator2.get_model()

        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

        assert set(model1_state_dict.keys()) == set(model2_state_dict.keys())

        for k in model1_state_dict:
            assert torch.equal(model1_state_dict[k], model2_state_dict[k])
        estimator2.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
