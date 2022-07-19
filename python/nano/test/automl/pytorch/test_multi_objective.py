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


import pytest
from unittest import TestCase

from bigdl.nano.pytorch import Trainer
import bigdl.nano.automl.hpo.space as space

import torch
from torch.utils.data import DataLoader, Dataset
from _helper import BoringModel, RandomDataset
import bigdl.nano.automl.hpo as hpo

class TestMOHPO(TestCase):

    def test_custom_mo_hpo(self):

        @hpo.plmodel()
        class CustomModel(BoringModel):
            """Customized Model."""
            def __init__(self,
                        out_dim1,
                        out_dim2,
                        dropout_1,
                        dropout_2,
                        learning_rate=0.1,
                        batch_size=16):
                super().__init__()
                layers = []
                input_dim = 32
                for out_dim, dropout in [(out_dim1, dropout_1),
                                        (out_dim2,dropout_2)]:
                    layers.append(torch.nn.Linear(input_dim, out_dim))
                    layers.append(torch.nn.Tanh())
                    layers.append(torch.nn.Dropout(dropout))
                    input_dim = out_dim
                layers.append(torch.nn.Linear(input_dim, 2))
                self.layers: torch.nn.Module = torch.nn.Sequential(*layers)
                self.save_hyperparameters()

            def configure_optimizers(self):
                # set learning rate in the optimizer
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                self.optimizer = torch.optim.Adam(self.layers.parameters(),
                                                  lr=self.hparams.learning_rate)
                return [self.optimizer], []

            def train_dataloader(self):
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

            def val_dataloader(self):
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

        model = CustomModel(
            out_dim1=space.Categorical(16,32),
            out_dim2=space.Categorical(16,32),
            dropout_1=space.Categorical(0.1, 0.2, 0.3, 0.4, 0.5),
            dropout_2 = space.Categorical(0.1,0.2),
            learning_rate = space.Real(0.001,0.01,log=True),
            batch_size = space.Categorical(32,64)
            )

        trainer = Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=2,
            use_hpo=True,
        )

        best_trials = trainer.search(
            model,
            target_metric=['val_loss', 'val_mae'],
            directions=['minimize', 'minimize'],
            n_trials=4,
            max_epochs=3,
        )

    def test_latency_hpo(self):

        @hpo.plmodel()
        class CustomModel(BoringModel):
            """Customized Model."""
            def __init__(self,
                        out_dim1,
                        out_dim2,
                        dropout_1,
                        dropout_2,
                        learning_rate=0.1,
                        batch_size=16):
                super().__init__()
                layers = []
                input_dim = 32
                for out_dim, dropout in [(out_dim1, dropout_1),
                                        (out_dim2,dropout_2)]:
                    layers.append(torch.nn.Linear(input_dim, out_dim))
                    layers.append(torch.nn.Tanh())
                    layers.append(torch.nn.Dropout(dropout))
                    input_dim = out_dim
                layers.append(torch.nn.Linear(input_dim, 2))
                self.layers: torch.nn.Module = torch.nn.Sequential(*layers)
                self.save_hyperparameters()

            def configure_optimizers(self):
                # set learning rate in the optimizer
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                self.optimizer = torch.optim.Adam(self.layers.parameters(),
                                                  lr=self.hparams.learning_rate)
                return [self.optimizer], []

            def train_dataloader(self):
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

            def val_dataloader(self):
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

        model = CustomModel(
            out_dim1=space.Categorical(16,32),
            out_dim2=space.Categorical(16,32),
            dropout_1=space.Categorical(0.1, 0.2, 0.3, 0.4, 0.5),
            dropout_2 = space.Categorical(0.1,0.2),
            learning_rate = space.Real(0.001,0.01,log=True),
            batch_size = space.Categorical(32,64)
            )

        trainer = Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=2,
            use_hpo=True,
        )
        
        best_trials = trainer.search(
            model,
            target_metric=['val_loss', 'latency'],
            directions=['minimize', 'minimize'],
            n_trials=4,
            max_epochs=3,
        )

    def test_invalid_input(self):
    
        @hpo.plmodel()
        class CustomModel(BoringModel):
            """Customized Model."""
            def __init__(self,
                        out_dim1,
                        out_dim2,
                        dropout_1,
                        dropout_2,
                        learning_rate=0.1,
                        batch_size=16):
                super().__init__()
                layers = []
                input_dim = 32
                for out_dim, dropout in [(out_dim1, dropout_1),
                                        (out_dim2,dropout_2)]:
                    layers.append(torch.nn.Linear(input_dim, out_dim))
                    layers.append(torch.nn.Tanh())
                    layers.append(torch.nn.Dropout(dropout))
                    input_dim = out_dim
                layers.append(torch.nn.Linear(input_dim, 2))
                self.layers: torch.nn.Module = torch.nn.Sequential(*layers)
                self.save_hyperparameters()

            def configure_optimizers(self):
                # set learning rate in the optimizer
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                self.optimizer = torch.optim.Adam(self.layers.parameters(),
                                                  lr=self.hparams.learning_rate)
                return [self.optimizer], []

            def train_dataloader(self):
                print("setting initial learning rate to",str(self.hparams.learning_rate))
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

            def val_dataloader(self):
                return DataLoader(RandomDataset(32, 64),
                                  batch_size=self.hparams.batch_size)

        model = CustomModel(
            out_dim1=space.Categorical(16,32),
            out_dim2=space.Categorical(16,32),
            dropout_1=space.Categorical(0.1, 0.2, 0.3, 0.4, 0.5),
            dropout_2 = space.Categorical(0.1,0.2),
            learning_rate = space.Real(0.001,0.01,log=True),
            batch_size = space.Categorical(32,64)
            )

        trainer = Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=2,
            use_hpo=True,
        )
        
        with pytest.raises(RuntimeError) as e:
            best_trials = trainer.search(
                model,
                target_metric=['val_loss', 'latency'],
                direction='minimize',
                n_trials=4,
                max_epochs=3,
            )
        error_msg = e.value.args[0]
        assert error_msg == "In multi-objective optimization, you must " \
                            "explicitly specify the direction for each metric"


if __name__ == '__main__':
    pytest.main([__file__])
