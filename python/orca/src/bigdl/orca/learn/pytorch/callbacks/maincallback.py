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
from .base import Callback
import torch
from bigdl.dllib.utils.log4Error import invalidInputError


def make_only_mainCallback(callbacks: list):
    _num_MCB = 0
    for i in range(len(callbacks)):
        if isinstance(callbacks[i], MainCallback):
            _num_MCB += 1
            # MainCallback should be called at the highest priority
            callbacks[0], callbacks[i] = callbacks[i], callbacks[0]

    if _num_MCB == 0:
        callbacks.insert(0, MainCallback())
    elif _num_MCB > 1:
        invalidInputError(False, f"Expect at most one MainCallback"
                          "instance to be passed to torch estimator, "
                          "got {_num_MCB} MainCallback instances.")


class MainCallback(Callback):
    """
    MainCallback is a one-of-a-kind callback that contains hook functions:
        - `on_iter_forward`
        - `on_iter_backward`
        - `on_lr_adjust`

    These methods are somewhat special, because only one special MainCallback
    should be allowed to implement these methods among all callbacks, otherwise
    there will propagate forward and backward twice.
    """
    def on_iter_forward(self, runner):
        """
        If `on_train_forward` and `on_val_forward` are not overridden,
        this will be called during forward when training and validating.
        Any behavior inconsistent with the default forward behavior should be overridden here.
        """
        # unpack features into features and targets
        *features, target = runner.batch
        # Forward features
        runner.output = runner.model(*features)
        # Ensure `targetL` and `outputL` are always in a list format.
        targetL = [target] if not isinstance(target, (list, tuple)) else target
        outputL = [runner.output] if not isinstance(runner.output, (list, tuple)) else runner.output
        # Compute loss
        runner.loss = runner.criterion(*outputL, *targetL)
        runner.target = target

    def on_iter_backward(self, runner):
        """
        this will be called during backward when training.
        Any behavior inconsistent with the default backward behavior should be overridden here.
        """
        runner.optimizer.zero_grad()
        runner.loss.backward()
        runner.optimizer.step()

    # TODO: Refactor scheduler update logic in TorchRunner
    def on_lr_adjust(self, runner):
        """
        this will be called during adjusting scheduler when each epoch ends.
        By default, this will step scheduler if there is scheduler in runner.
        Any behavior inconsistent with the default behavior should be overridden here.
        """
        if runner.scheduler:
            runner.scheduler.step()

    def on_train_forward(self, runner):
        """
        Called during training.
        Any behavior inconsistent with the default training behavior should be overridden here.
        """
        self.on_iter_forward(runner)

    def on_val_forward(self, runner):
        """
        Called during validate.
        Any behavior inconsistent with the default training behavior should be overridden here.
        """
        self.on_iter_forward(runner)

    def on_pred_forward(self, runner):
        """
        Called during prediction.
        Any behavior inconsistent with the default prediction behavior should be overridden here.
        """
        output = runner.model(*runner.batch)

        if len(output.size()) > 1:
            # In case there is extra trailing dimensions.
            for i in reversed(range(1, len(output.size()))):
                output = torch.squeeze(output, i)

        # todo support multi-output model
        runner.output = output.detach().numpy()
