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
# ==============================================================================
"""SparseAdam optimizer pytorch implementation."""

# This file is adapted from Lazy Adam optimizer (PyTorch).
# https://github.com/davda54/lazy-adam/blob/master/lazy_adam.py

# MIT License
#
# Copyright (c) 2020 David Samuel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import math
import torch
from torch.optim.optimizer import Optimizer
from bigdl.nano.utils.log4Error import invalidInputError


class SparseAdam(Optimizer):
    """
    A variant of the Adam optimizer that can handles both sparse and non-sparse updates.

    The original Adam algorithm maintains two moving-average accumulators for
    each trainable variable; the accumulators are updated at every step.
    This class provides lazier handling of gradient updates for sparse
    variables.  It only updates moving-average accumulators for sparse variable
    indices that appear in the current batch, rather than updating the
    accumulators for all indices. Compared with the original Adam optimizer,
    it can provide large improvements in model training throughput for some
    applications. However, it provides slightly different semantics than the
    original Adam algorithm, and may lead to different empirical results.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Construct a new SparseAdam optimizer.

        param lr: A `Tensor` or a floating point value. or a schedule
            that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
            The learning rate.
        param beta_1: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 1st moment estimates.
        param beta_2: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 2nd moment estimates.
        param epsilon: A small constant for numerical stability.
            This epsilon is "epsilon hat" in
            [Adam: A Method for Stochastic Optimization. Kingma et al., 2014]
            (http://arxiv.org/abs/1412.6980) (in the formula just
            before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        """
        if not 0.0 < lr:
            invalidInputError(False, "Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            invalidInputError(False, "Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            invalidInputError(False,
                              "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            invalidInputError(False,
                              "Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        :param closure: A optional callable. A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    self._sparse_step(group, p, grad)
                else:
                    self._dense_step(group, p, grad)

        return loss

    def _sparse_step(self, group, param, grad):
        state = self.state[param]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param.data)

        state['step'] += 1

        grad = grad.coalesce()  # the update is non-linear so indices must be unique
        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new
        # <==> old += (1 - b) * (new - old)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # Dense addition again is intended, avoiding another sparse_mask
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
        del exp_avg_update_values, exp_avg_sq_update_values

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        param.data.add_(make_sparse(-step_size * numer.div_(denom)))

    def _dense_step(self, group, param, grad):
        state = self.state[param]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        param.data.addcdiv_(-step_size, exp_avg, denom)
