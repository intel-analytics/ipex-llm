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
# this file is adapted from
# https://github.com/rnwzd/FSPBT-Image-Translation/blob/master/train.py

# MIT License

# Copyright (c) 2022 Lorenzo Breschi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch

from litmodels import LitModel
from data import PatchDataModule, prepare_data

data_path = Path('data/')
# data is excpected to be in folders:
# data_path /
#           input
#           target
#           mask (optional)

model_save_path = data_path / 'models'

if __name__ == "__main__":

    logger = TensorBoardLogger(Path(), 'lightning_logs')

    profiler = pl.profiler.SimpleProfiler()

    callbacks = []

    train_image_dd = prepare_data(data_path)

    dm = PatchDataModule(train_image_dd,
                         patch_size=2**6,
                         batch_size=2**3,
                         patch_num=2**6)

    model = LitModel( use_adversarial=True)

    # uncomment next line to start from latest checkpoint
    # model = LitModel.load_from_checkpoint(model_save_path/"latest.ckpt")
    from bigdl.nano.pytorch import Trainer

    trainer = Trainer(
        gpus=0,
        max_epochs=100,
        log_every_n_steps=8,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        check_val_every_n_epoch=20,
        reload_dataloaders_every_n_epochs=1,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        # fast_dev_run=True,
    )

    import time
    st = time.perf_counter()
    trainer.fit(model, dm)
    end = time.perf_counter()
    print(f"training cost {end-st}s.")

    trainer.save_checkpoint(model_save_path/"latest.ckpt")
    torch.save(model.generator, model_save_path/"generator.pt")
    torch.save(model.discriminator, model_save_path/"discriminator.pt")
