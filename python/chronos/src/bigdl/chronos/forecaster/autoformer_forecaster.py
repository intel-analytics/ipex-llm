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
from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.model.autoformer.Autoformer import AutoFormer
import torch.nn as nn
import numpy as np
import os


class AutoformerForecaster(BasePytorchForecaster):
    def __init__(self,
                 seq_len,
                 label_len,
                 pred_len,
                 enc_in,
                 freq,
                 dec_in,
                 c_out,
                 output_attention=False,
                 moving_avg=25,
                 d_model=512,
                 embed='timeF',
                 dropout=0.05,
                 factor=3,
                 n_head=8,
                 d_ff=2048,
                 activation='gelu',
                 e_layer=2,
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

        self.distributed = False
        self.seed = None
        self.checkpoint_callback = False
        current_num_threads = torch.get_num_threads()
        self.num_processes = max(1, current_num_threads//8)
        self.use_ipex = False
        self.onnx_available = True
        self.quantize_available = True
        self.use_amp = False

        self.model_creator = model_creator
        self.internal = model_creator(self.config)

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

    def load(self, checkpoint_file, quantize_checkpoint_file=None):
        """
        restore the forecaster.
        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        :param quantize_checkpoint_file: The checkpoint file location you want to
               load the quantized forecaster.
        """
        from bigdl.chronos.pytorch import TSTrainer as Trainer

        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
            from bigdl.chronos.pytorch import TSTrainer as Trainer

            model = self.model_creator(self.config)
            self.internal = LightningModuleFromTorch.load_from_checkpoint(checkpoint_file,
                                                                          model=model,
                                                                          )
            self.internal = Trainer.compile(self.internal)
            self.fitted = True
            if quantize_checkpoint_file:
                # self.internal.load_quantized_state_dict(torch.load(quantize_checkpoint_file))
                self.pytorch_int8 = Trainer.load(quantize_checkpoint_file,
                                                 self.internal)
            # This trainer is only for quantization, once the user call `fit`, it will be
            # replaced according to the new training config
            self.trainer = Trainer(logger=False, max_epochs=1,
                                   checkpoint_callback=self.checkpoint_callback,
                                   num_processes=self.num_processes, use_ipex=self.use_ipex)

    def evaluate(self, batch):
        total_loss = []
        self.internal.eval()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch):
            batch_x = batch_x.float()
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            dec_inp = torch.zeros_like(batch_y[:, -self.config['pred_len']:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config['label_len'], :], dec_inp], dim=1).float()

            outputs = self.internal(batch_x, batch_x_mark, dec_inp,
                                    batch_y_mark, batch_y)

            outputs = outputs[:, -self.config['pred_len']:, :]
            batch_y = batch_y[:, -self.config['pred_len']:, :]

            loss = nn.MSELoss()

            loss_result = loss(outputs, batch_y)
            total_loss.append(loss_result)
        total_loss = torch.mean(torch.stack(total_loss))
        return total_loss

    def predict(self, data, load=False):
        preds = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data):
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0],
                                  self.config['pred_len'], batch_y.shape[2]]).float()
            dec_inp = torch.cat([batch_y[:, :self.config['label_len'], :], dec_inp],
                                dim=1).float()
            # encoder - decoder
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if self.config['output_attention']:
                        outputs = self.internal(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.internal(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.config['output_attention']:
                    outputs = self.internal(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.internal(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            preds.append(outputs)

        preds = np.array(preds)

        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
