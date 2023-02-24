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
import os
import shutil
import tempfile

from bigdl.orca.learn.pytorch.callbacks import Callback
from bigdl.dllib.utils.log4Error import invalidInputError

tqdm = None
try:
    from tqdm import tqdm
except ImportError:
    pass


def is_tqdm_exists(callbacks):
    for callback in callbacks:
        if isinstance(callback, TqdmCallback):
            return True
    return False


class TqdmCallback(Callback):

    def __init__(self):
        self._progress_bar = None
        super().__init__()

    def after_train_iter(self, runner):
        if self._is_rank_zero(runner):
            runner._progress_bar.n = runner.batch_idx
            postfix = {}
            if "train_loss" in runner.metrics_stats:
                postfix.update(loss=runner.metrics_stats["train_loss"])
            runner._progress_bar.set_postfix(postfix)

    def before_train_epoch(self, runner):
        if self._is_rank_zero(runner):
            desc = "{}/{}e".format(runner.epochs + 1,
                                   runner.num_epochs)

            invalidInputError(tqdm is not None,
                              "tqdm is not installed, please install with 'pip install tqdm'")
            runner._progress_bar = tqdm(total=len(runner.train_loader),
                                        desc=desc,
                                        unit="batch",
                                        leave=False)

    def after_val_iter(self, runner):
        if self._is_rank_zero(runner):
            runner._progress_bar.n = runner.batch_idx
            postfix = {}
            postfix.update(loss=runner.loss.item())
            runner._progress_bar.set_postfix(postfix)

    def before_val_epoch(self, runner):
        if self._is_rank_zero(runner):
            desc = "1/1e"

            invalidInputError(tqdm is not None,
                              "tqdm is not installed, please install with 'pip install tqdm'")
            runner._progress_bar = tqdm(total=len(runner.val_loader),
                                        desc=desc,
                                        unit="batch",
                                        leave=False)

    def after_pred_iter(self, runner):
        if self._is_rank_zero(runner):
            print("\r Predict batch_idx: {%d}" % runner.batch_idx, end="")

    def _is_rank_zero(self, runner):
        invalidInputError(runner, "Sanity check failed. Runner must not be None!")
        rank = runner.rank
        return rank == 0
