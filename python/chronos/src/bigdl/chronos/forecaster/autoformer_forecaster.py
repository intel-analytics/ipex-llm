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

import torch
from bigdl.chronos.forecaster.base_forecaster import BasePytorchForecaster
from bigdl.chronos.model.autoformer import model_creator


class AutoformerForecaster(BasePytorchForecaster):
    def __init__(self,
                 seq_len,
                 label_len,
                 pred_len,
                 output_attention=False,
                 moving_avg=25,
                 enc_in,
                 d_model=512,
                 embed='timeF',
                 freq,
                 dropout=0.05,
                 dec_in,
                 factor=3,
                 n_head=8,
                 d_ff=2048,
                 activation='gelu',
                 e_layer=2,
                 c_out,
                 d_layers=1):
        # config setting
        self.config = {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "output_attention": output_attention,
            "moving_avg": moving_avg,
            "enc_in": enc_in,
            "d_model": d_model,
            "embed": embed,
            "freq": freq,
            "dropout": dropout,
            "dec_in": dec_in,
            "factor": factor,
            "n_head": n_head,
            "d_ff": d_ff,
            "activation": activation,
            "e_layer": e_layer,
            "c_out": c_out,
            "d_layers": d_layers
        }

        self.model_creator = model_creator

        super().__init__()

    def fit(self, data, epochs=1, batch_size=32):
        # input transform
        if isinstance(data, DataLoader) and self.distributed:
            data = loader_to_creator(data)
        if isinstance(data, tuple) and self.distributed:
            data = np_to_creator(data)
        try:
            from bigdl.orca.data.shard import SparkXShards
            if isinstance(data, SparkXShards) and not self.distributed:
                warnings.warn("Xshards is collected to local since the "
                              "forecaster is non-distribued.")
                data = xshard_to_np(data)
        except ImportError:
            pass

        # fit on internal
        if self.distributed:
            # for cluster mode
            from bigdl.orca.common import OrcaContext
            sc = OrcaContext.get_spark_context().getConf()
            num_nodes = 1 if sc.get('spark.master').startswith('local') \
                else int(sc.get('spark.executor.instances'))
            if batch_size % self.workers_per_node != 0:
                raise RuntimeError("Please make sure that batch_size can be divisible by "
                                   "the product of worker_per_node and num_nodes, "
                                   f"but 'batch_size' is {batch_size}, 'workers_per_node' "
                                   f"is {self.workers_per_node}, 'num_nodes' is {num_nodes}")
            batch_size //= (self.workers_per_node * num_nodes)
            return self.internal.fit(data=data,
                                     epochs=epochs,
                                     batch_size=batch_size)
        else:
            from bigdl.chronos.pytorch import TSTrainer as Trainer

            # # numpy data shape checking
            # if isinstance(data, tuple):
            #     check_data(data[0], data[1], self.data_config)
            # else:
            #     warnings.warn("Data shape checking is not supported by dataloader input.")

            # data transformation
            if isinstance(data, tuple):
                data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                                torch.from_numpy(data[1]),
                                                torch.from_numpy(data[2]),
                                                torch.from_numpy(data[3]),),
                                  batch_size=batch_size,
                                  shuffle=True)

            # Trainer init and fitting
            self.trainer = Trainer(max_epochs=epochs)
            self.trainer.fit(self.internal, data)
            self.fitted = True
