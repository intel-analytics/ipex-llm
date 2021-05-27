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
# This file is adapted from the DeepGlo Project. https://github.com/rajatsen91/deepglo
#
# Note: This license has also been called the "New BSD License" or "Modified BSD License". See also
# the 2-clause BSD License.
#
# Copyright (c) 2019 The DeepGLO Project.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from __future__ import print_function
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from zoo.chronos.model.tcmf.data_loader import TCMFDataLoader
from zoo.chronos.model.tcmf.local_model import TemporalConvNet, LocalModel
from zoo.chronos.model.tcmf.time import TimeCovariates

import copy

import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


def get_model(A, y, lamb=0):
    """
    Regularized least-squares
    """
    n_col = A.shape[1]
    return np.linalg.lstsq(
        A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
    )


class DeepGLO(object):
    def __init__(
        self,
        vbsize=150,
        hbsize=256,
        num_channels_X=[32, 32, 32, 32, 1],
        num_channels_Y=[32, 32, 32, 32, 1],
        kernel_size=7,
        dropout=0.2,
        rank=64,
        kernel_size_Y=7,
        lr=0.0005,
        normalize=False,
        use_time=True,
        svd=False,
        forward_cov=False,
    ):

        self.use_time = use_time
        self.dropout = dropout
        self.forward_cov = forward_cov
        self.Xseq = TemporalConvNet(
            num_inputs=1,
            num_channels=num_channels_X,
            kernel_size=kernel_size,
            dropout=dropout,
            init=True,
        )
        self.vbsize = vbsize
        self.hbsize = hbsize
        self.num_channels_X = num_channels_X
        self.num_channels_Y = num_channels_Y
        self.kernel_size_Y = kernel_size_Y
        self.rank = rank
        self.kernel_size = kernel_size
        self.lr = lr
        self.normalize = normalize
        self.svd = svd

    def tensor2d_to_temporal(self, T):
        T = T.view(1, T.size(0), T.size(1))
        T = T.transpose(0, 1)
        return T

    def temporal_to_tensor2d(self, T):
        T = T.view(T.size(0), T.size(2))
        return T

    def calculate_newX_loss_vanilla(self, Xn, Fn, Yn, Xf, alpha):
        Yout = torch.mm(Fn, Xn)
        cr1 = nn.L1Loss()
        cr2 = nn.MSELoss()
        l1 = cr2(Yout, Yn) / torch.mean(Yn ** 2)
        l2 = cr2(Xn, Xf) / torch.mean(Xf ** 2)
        return (1 - alpha) * l1 + alpha * l2

    def recover_future_X(
        self,
        last_step,
        future,
        num_epochs=50,
        alpha=0.5,
        vanilla=True,
        tol=1e-7,
    ):
        rg = max(
            1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels_X) - 1),
            1 + 2 * (self.kernel_size_Y - 1) * 2 ** (len(self.num_channels_Y) - 1),
        )
        X = self.X[:, last_step - rg: last_step]
        X = self.tensor2d_to_temporal(X)
        outX = self.predict_future(model=self.Xseq, inp=X, future=future)
        outX = self.temporal_to_tensor2d(outX)
        Xf = outX[:, -future::]
        Yn = self.Ymat[:, last_step: last_step + future]
        Yn = torch.from_numpy(Yn).float()

        Fn = self.F

        Xt = torch.zeros(self.rank, future).float()
        Xn = torch.normal(Xt, 0.1)

        lprev = 0
        for i in range(num_epochs):
            Xn = Variable(Xn, requires_grad=True)
            optim_Xn = optim.Adam(params=[Xn], lr=self.lr)
            optim_Xn.zero_grad()
            loss = self.calculate_newX_loss_vanilla(
                Xn, Fn.detach(), Yn.detach(), Xf.detach(), alpha
            )
            loss.backward()
            optim_Xn.step()
            # Xn = torch.clamp(Xn.detach(), min=0)

            if np.abs(lprev - loss.item()) <= tol:
                break

            if i % 1000 == 0:
                print(f"Recovery Loss of epoch {i} is: " + str(loss.item()))
                lprev = loss.item()

        return Xn.detach()

    def step_factX_loss(self, inp, out, last_vindex, last_hindex, reg=0.0):
        Xout = self.X[:, last_hindex + 1: last_hindex + 1 + out.size(2)]
        Fout = self.F[self.D.I[last_vindex: last_vindex + out.size(0)], :]
        Xout = Variable(Xout, requires_grad=True)
        out = self.temporal_to_tensor2d(out)
        optim_X = optim.Adam(params=[Xout], lr=self.lr)
        Hout = torch.matmul(Fout, Xout)
        optim_X.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(Xout, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_X.step()
        # Xout = torch.clamp(Xout, min=0)
        self.X[:, last_hindex + 1: last_hindex + 1 + inp.size(2)] = Xout.detach()
        return loss

    def step_factF_loss(self, inp, out, last_vindex, last_hindex, reg=0.0):
        Xout = self.X[:, last_hindex + 1: last_hindex + 1 + out.size(2)]
        Fout = self.F[self.D.I[last_vindex: last_vindex + out.size(0)], :]
        Fout = Variable(Fout, requires_grad=True)
        optim_F = optim.Adam(params=[Fout], lr=self.lr)
        out = self.temporal_to_tensor2d(out)
        Hout = torch.matmul(Fout, Xout)
        optim_F.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(Fout, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_F.step()
        self.F[
            self.D.I[last_vindex: last_vindex + inp.size(0)], :
        ] = Fout.detach()
        return loss

    def step_temporal_loss_X(self, inp, last_vindex, last_hindex):
        Xin = self.X[:, last_hindex: last_hindex + inp.size(2)]
        Xout = self.X[:, last_hindex + 1: last_hindex + 1 + inp.size(2)]
        for p in self.Xseq.parameters():
            p.requires_grad = False
        Xin = Variable(Xin, requires_grad=True)
        Xout = Variable(Xout, requires_grad=True)
        optim_out = optim.Adam(params=[Xout], lr=self.lr)
        Xin = self.tensor2d_to_temporal(Xin)
        Xout = self.tensor2d_to_temporal(Xout)
        hatX = self.Xseq(Xin)
        optim_out.zero_grad()
        loss = torch.mean(torch.pow(Xout - hatX.detach(), 2))
        loss.backward()
        optim_out.step()
        # Xout = torch.clamp(Xout, min=0)
        temp = self.temporal_to_tensor2d(Xout.detach())
        self.X[:, last_hindex + 1: last_hindex + 1 + inp.size(2)] = temp
        return loss

    def predict_future_batch(self, model, inp, future=10):
        out = model(inp)
        output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
        out = torch.cat((inp, output), dim=2)
        for i in range(future - 1):
            inp = out
            out = model(inp)
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            out = torch.cat((inp, output), dim=2)

        out = self.temporal_to_tensor2d(out)
        out = np.array(out.detach())
        return out

    def predict_future(self, model, inp, future=10, bsize=90):
        n = inp.size(0)
        ids = np.arange(0, n, bsize)
        ids = list(ids) + [n]
        out = self.predict_future_batch(model, inp[ids[0]: ids[1], :, :], future)

        for i in range(1, len(ids) - 1):
            temp = self.predict_future_batch(
                model, inp[ids[i]: ids[i + 1], :, :], future
            )
            out = np.vstack([out, temp])

        out = torch.from_numpy(out).float()
        return self.tensor2d_to_temporal(out)

    def predict_global(
        self, ind, last_step=100, future=10, normalize=False, bsize=90
    ):

        if ind is None:
            ind = np.arange(self.Ymat.shape[0])

        self.Xseq = self.Xseq.eval()

        rg = max(
            1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels_X) - 1),
            1 + 2 * (self.kernel_size_Y - 1) * 2 ** (len(self.num_channels_Y) - 1),
        )
        X = self.X[:, last_step - rg: last_step]
        n = X.size(0)
        T = X.size(1)
        X = self.tensor2d_to_temporal(X)
        outX = self.predict_future(
            model=self.Xseq, inp=X, future=future, bsize=bsize
        )

        outX = self.temporal_to_tensor2d(outX)

        F = self.F

        Y = torch.matmul(F, outX)

        Y = np.array(Y[ind, :].detach())

        del F

        for p in self.Xseq.parameters():
            p.requires_grad = True

        if normalize:
            Y = Y - self.mini
            Y = Y * self.s[ind, None] + self.m[ind, None]
            return Y
        else:
            return Y

    def train_Xseq(self, Ymat, num_epochs=20, val_len=24, early_stop=False, tenacity=3):
        seq = self.Xseq
        num_channels = self.num_channels_X
        kernel_size = self.kernel_size
        vbsize = min(self.vbsize, Ymat.shape[0] / 2)

        for p in seq.parameters():
            p.requires_grad = True

        TC = LocalModel(
            Ymat=Ymat,
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            vbsize=vbsize,
            hbsize=self.hbsize,
            normalize=False,
            end_index=self.end_index - val_len,
            val_len=val_len,
            lr=self.lr,
        )

        TC.train_model(num_epochs=num_epochs, early_stop=early_stop, tenacity=tenacity)

        self.Xseq = TC.seq

    def train_factors(
        self,
        reg_X=0.0,
        reg_F=0.0,
        mod=5,
        val_len=24,
        early_stop=False,
        tenacity=3,
        ind=None,
        seed=False,
    ):
        self.D.epoch = 0
        self.D.vindex = 0
        self.D.hindex = 0
        for p in self.Xseq.parameters():
            p.requires_grad = True

        l_F = [0.0]
        l_X = [0.0]
        l_X_temporal = [0.0]
        iter_count = 0
        vae = float("inf")
        scount = 0
        Xbest = self.X.clone()
        Fbest = self.F.clone()
        while self.D.epoch < self.num_epochs:
            last_epoch = self.D.epoch
            last_vindex = self.D.vindex
            last_hindex = self.D.hindex
            inp, out, vindex, hindex = self.D.next_batch()

            step_l_F = self.step_factF_loss(inp, out, last_vindex, last_hindex, reg=reg_F)
            l_F = l_F + [step_l_F.item()]
            step_l_X = self.step_factX_loss(inp, out, last_vindex, last_hindex, reg=reg_X)
            l_X = l_X + [step_l_X.item()]
            if seed is False and iter_count % mod == 1:
                l2 = self.step_temporal_loss_X(inp, last_vindex, last_hindex)
                l_X_temporal = l_X_temporal + [l2.item()]
            iter_count = iter_count + 1

            if self.D.epoch > last_epoch:
                print("Entering Epoch#{}".format(self.D.epoch))
                print("Factorization Loss F:{}".format(np.mean(l_F)))
                print("Factorization Loss X:{}".format(np.mean(l_X)))
                print("Temporal Loss X:{}".format(np.mean(l_X_temporal)))

                if ind is None:
                    ind = np.arange(self.Ymat.shape[0])
                else:
                    ind = ind
                inp = self.predict_global(
                    ind,
                    last_step=self.end_index - val_len,
                    future=val_len,
                )
                R = self.Ymat[ind, self.end_index - val_len: self.end_index]
                S = inp[:, -val_len::]
                ve = np.abs(R - S).mean() / np.abs(R).mean()
                # print("Validation Loss (Global): ", ve)
                print("Validation Loss (Global):{}".format(ve))
                if ve <= vae:
                    vae = ve
                    scount = 0
                    Xbest = self.X.clone()
                    Fbest = self.F.clone()
                    # Xseqbest = TemporalConvNet(
                    #     num_inputs=1,
                    #     num_channels=self.num_channels_X,
                    #     kernel_size=self.kernel_size,
                    #     dropout=self.dropout,
                    # )
                    # Xseqbest.load_state_dict(self.Xseq.state_dict())
                    Xseqbest = pickle.loads(pickle.dumps(self.Xseq))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        # print("Early Stopped")
                        print("Early Stopped")
                        self.X = Xbest
                        self.F = Fbest
                        self.Xseq = Xseqbest
                        break

    def create_Ycov(self):
        t0 = self.end_index + 1
        self.D.epoch = 0
        self.D.vindex = 0
        self.D.hindex = 0
        Ycov = copy.deepcopy(self.Ymat[:, 0:t0])
        Ymat_now = self.Ymat[:, 0:t0]

        self.Xseq = self.Xseq.eval()

        while self.D.epoch < 1:
            last_epoch = self.D.epoch
            last_vindex = self.D.vindex
            last_hindex = self.D.hindex
            inp, out, vindex, hindex = self.D.next_batch()

            Xin = self.tensor2d_to_temporal(self.X[:, last_hindex: last_hindex + inp.size(2)])
            Xout = self.temporal_to_tensor2d(self.Xseq(Xin))
            Fout = self.F[self.D.I[last_vindex: last_vindex + out.size(0)], :]
            output = np.array(torch.matmul(Fout, Xout).detach())
            Ycov[
                last_vindex: last_vindex + output.shape[0],
                last_hindex + 1: last_hindex + 1 + output.shape[1],
            ] = output

        for p in self.Xseq.parameters():
            p.requires_grad = True

        if self.period is None:
            Ycov_wc = np.zeros(shape=[Ycov.shape[0], 1, Ycov.shape[1]])
            if self.forward_cov:
                Ycov_wc[:, 0, 0:-1] = Ycov[:, 1::]
            else:
                Ycov_wc[:, 0, :] = Ycov
        else:
            Ycov_wc = np.zeros(shape=[Ycov.shape[0], 2, Ycov.shape[1]])
            if self.forward_cov:
                Ycov_wc[:, 0, 0:-1] = Ycov[:, 1::]
            else:
                Ycov_wc[:, 0, :] = Ycov
            Ycov_wc[:, 1, self.period - 1::] = Ymat_now[:, 0: -(self.period - 1)]
        return Ycov_wc

    def train_Yseq(self, num_epochs=20,
                   covariates=None,
                   dti=None,
                   val_len=24,
                   num_workers=1,
                   ):
        Ycov = self.create_Ycov()
        self.Yseq = LocalModel(
            self.Ymat,
            num_inputs=1,
            num_channels=self.num_channels_Y,
            kernel_size=self.kernel_size_Y,
            dropout=self.dropout,
            vbsize=self.vbsize,
            hbsize=self.hbsize,
            lr=self.lr,
            val_len=val_len,
            test=True,
            end_index=self.end_index - val_len,
            normalize=False,
            start_date=self.start_date,
            freq=self.freq,
            covariates=covariates,
            use_time=self.use_time,
            dti=dti,
            Ycov=Ycov,
        )
        val_loss = self.Yseq.train_model(num_epochs=num_epochs,
                                         num_workers=num_workers,
                                         early_stop=False)
        return val_loss

    def train_all_models(
            self,
            Ymat,
            val_len=24,
            start_date="2016-1-1",
            freq="H",
            covariates=None,
            dti=None,
            period=None,
            init_epochs=100,
            alt_iters=10,
            y_iters=200,
            tenacity=7,
            mod=5,
            max_FX_epoch=300,
            max_TCN_epoch=300,
            num_workers=1,
    ):
        self.end_index = Ymat.shape[1]
        self.start_date = start_date
        self.freq = freq
        self.period = period
        self.covariates = covariates
        self.dti = dti

        if self.normalize:
            self.s = np.std(Ymat[:, 0:self.end_index], axis=1)
            # self.s[self.s == 0] = 1.0
            self.s += 1.0
            self.m = np.mean(Ymat[:, 0:self.end_index], axis=1)
            self.Ymat = (Ymat - self.m[:, None]) / self.s[:, None]
            self.mini = np.abs(np.min(self.Ymat))
            self.Ymat = self.Ymat + self.mini
        else:
            self.Ymat = Ymat

        n, T = self.Ymat.shape
        t0 = self.end_index + 1
        if t0 > T:
            self.Ymat = np.hstack([self.Ymat, self.Ymat[:, -1].reshape(-1, 1)])
        if self.svd:
            indices = np.random.choice(self.Ymat.shape[0], self.rank, replace=False)
            X = self.Ymat[indices, 0:t0]
            mX = np.std(X, axis=1)
            mX[mX == 0] = 1.0
            X = X / mX[:, None]
            Ft = get_model(X.transpose(), self.Ymat[:, 0:t0].transpose(), lamb=0.1)
            F = Ft[0].transpose()
            self.X = torch.from_numpy(X).float()
            self.F = torch.from_numpy(F).float()
        else:
            R = torch.zeros(self.rank, t0).float()
            X = torch.normal(R, 0.1)
            C = torch.zeros(n, self.rank).float()
            F = torch.normal(C, 0.1)
            self.X = X.float()
            self.F = F.float()

        self.D = TCMFDataLoader(
            Ymat=self.Ymat,
            vbsize=self.vbsize,
            hbsize=self.hbsize,
            end_index=self.end_index,
            val_len=val_len,
            shuffle=False,
        )
        # print("-"*50+"Initializing Factors.....")
        logger.info("Initializing Factors")
        self.num_epochs = init_epochs
        self.train_factors(val_len=val_len)

        if alt_iters % 2 == 1:
            alt_iters += 1

        # print("Starting Alternate Training.....")
        logger.info("Starting Alternate Training.....")

        for i in range(1, alt_iters):
            if i % 2 == 0:
                logger.info("Training Factors. Iter#:{}".format(i))
                self.num_epochs = max_FX_epoch
                self.train_factors(
                    seed=False, val_len=val_len,
                    early_stop=True, tenacity=tenacity, mod=mod
                )
            else:
                # logger.info(
                #     "--------------------------------------------Training Xseq Model. Iter#:{}"
                #     .format(i)
                #     + "-------------------------------------------------------"
                # )
                logger.info("Training Xseq Model. Iter#:{}".format(i))

                self.num_epochs = max_TCN_epoch
                T = np.array(self.X.detach())
                self.train_Xseq(
                    Ymat=T,
                    num_epochs=self.num_epochs,
                    val_len=val_len,
                    early_stop=True,
                    tenacity=tenacity,
                )

        logger.info("Start training Yseq.....")
        val_loss = self.train_Yseq(num_epochs=y_iters,
                                   covariates=covariates,
                                   dti=dti,
                                   val_len=val_len,
                                   num_workers=num_workers,
                                   )
        return val_loss

    def append_new_y(self, Ymat_new, covariates_new=None, dti_new=None):
        # update Yseq
        # normalize the incremented Ymat if needed
        if self.normalize:
            Ymat_new = (Ymat_new - self.m[:, None]) / self.s[:, None]
            Ymat_new = Ymat_new + self.mini

        # append the new Ymat onto the original, note that self.end_index equals to the no.of time
        # steps of the original.
        n, T_added = Ymat_new.shape
        self.Ymat = np.concatenate((self.Ymat[:, : self.end_index], Ymat_new), axis=1)
        self.end_index = self.end_index + T_added

        n, T = self.Ymat.shape
        t0 = self.end_index + 1
        if t0 > T:
            self.Ymat = np.hstack([self.Ymat, self.Ymat[:, -1].reshape(-1, 1)])

        # update Yseq.covariates
        last_step = self.end_index - T_added
        new_covariates = self.get_future_time_covs(T_added, last_step,
                                                   future_covariates=covariates_new,
                                                   future_dti=dti_new)
        self.Yseq.covariates = np.hstack([self.Yseq.covariates[:, :last_step], new_covariates])

    def inject_new(self,
                   Ymat_new,
                   covariates_new=None,
                   dti_new=None):
        if self.Ymat.shape[0] != Ymat_new.shape[0]:
            raise ValueError("Expected incremental input with {} time series, got {} instead."
                             .format(self.Ymat.shape[0], Ymat_new.shape[0]))
        self.append_new_y(Ymat_new, covariates_new=covariates_new, dti_new=dti_new)
        n, T = self.Ymat.shape
        rank, XT = self.X.shape
        future = T - XT
        Xn = self.recover_future_X(
            last_step=XT,
            future=future,
            num_epochs=100000,
            alpha=0.3,
            vanilla=True,
        )
        self.X = torch.cat([self.X, Xn], dim=1)

    def get_time_covs(self, future_start_date, num_ts, future_covariates, future_dti):
        if self.use_time:
            future_time = TimeCovariates(
                start_date=future_start_date,
                freq=self.freq,
                normalized=True,
                num_ts=num_ts
            )
            if future_dti is not None:
                future_time.dti = future_dti
            time_covariates = future_time.get_covariates()
            if future_covariates is None:
                covariates = time_covariates
            else:
                covariates = np.vstack([time_covariates, future_covariates])
        else:
            covariates = future_covariates
        return covariates

    def get_future_time_covs(self, horizon, last_step, future_covariates, future_dti):
        if self.freq[0].isalpha():
            freq = "1" + self.freq
        else:
            freq = self.freq
        future_start_date = pd.Timestamp(self.start_date) + pd.Timedelta(freq) * last_step
        covs_future = self.get_time_covs(future_start_date=future_start_date,
                                         num_ts=horizon,
                                         future_covariates=future_covariates,
                                         future_dti=future_dti)
        return covs_future

    def get_prediction_time_covs(self, rg, horizon, last_step, future_covariates, future_dti):
        covs_past = self.Yseq.covariates[:, last_step - rg: last_step]
        covs_future = self.get_future_time_covs(horizon, last_step, future_covariates, future_dti)
        covs = np.concatenate([covs_past, covs_future], axis=1)
        return covs

    def predict_horizon(
            self,
            ind=None,
            future=10,
            future_covariates=None,
            future_dti=None,
            bsize=90,
            num_workers=1,
    ):
        last_step = self.end_index
        if ind is None:
            ind = np.arange(self.Ymat.shape[0])

        self.Yseq.seq = self.Yseq.seq.eval()
        self.Xseq = self.Xseq.eval()

        rg = max(
            1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels_X) - 1),
            1 + 2 * (self.kernel_size_Y - 1) * 2 ** (len(self.num_channels_Y) - 1),
        )
        covs = self.get_prediction_time_covs(rg, future, last_step, future_covariates, future_dti)

        yc = self.predict_global(
            ind=ind,
            last_step=last_step,
            future=future,
            normalize=False,
            bsize=bsize,
        )

        if self.period is None:
            ycovs = np.zeros(shape=[yc.shape[0], 1, yc.shape[1]])
            if self.forward_cov:
                ycovs[:, 0, 0:-1] = yc[:, 1::]
            else:
                ycovs[:, 0, :] = yc
        else:
            ycovs = np.zeros(shape=[yc.shape[0], 2, yc.shape[1]])
            if self.forward_cov:
                ycovs[:, 0, 0:-1] = yc[:, 1::]
            else:
                ycovs[:, 0, :] = yc
            period = self.period
            while last_step + future - (period - 1) > last_step + 1:
                period += self.period
            # The last coordinate is not used.
            ycovs[:, 1, period - 1::] = self.Ymat[
                :, last_step - rg: last_step + future - (period - 1)]

        Y = self.Yseq.predict_future(
            data_in=self.Ymat[ind, last_step - rg: last_step],
            covariates=covs,
            ycovs=ycovs,
            future=future,
            bsize=bsize,
            normalize=False,
            num_workers=num_workers,
        )

        if self.normalize:
            Y = Y - self.mini
            Y = Y * self.s[ind, None] + self.m[ind, None]
            return Y
        else:
            return Y

    def predict(
        self, ind=None, last_step=100, future=10, normalize=False, bsize=90
    ):

        if ind is None:
            ind = np.arange(self.Ymat.shape[0])

        self.Xseq = self.Xseq

        self.Yseq.seq = self.Yseq.seq.eval()
        self.Xseq = self.Xseq.eval()

        rg = max(
            1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels_X) - 1),
            1 + 2 * (self.kernel_size_Y - 1) * 2 ** (len(self.num_channels_Y) - 1),
        )
        covs = self.Yseq.covariates[:, last_step - rg: last_step + future]
        # print(covs.shape)
        yc = self.predict_global(
            ind=ind,
            last_step=last_step,
            future=future,
            normalize=False,
            bsize=bsize,
        )
        if self.period is None:
            ycovs = np.zeros(shape=[yc.shape[0], 1, yc.shape[1]])
            if self.forward_cov:
                ycovs[:, 0, 0:-1] = yc[:, 1::]
            else:
                ycovs[:, 0, :] = yc
        else:
            ycovs = np.zeros(shape=[yc.shape[0], 2, yc.shape[1]])
            if self.forward_cov:
                ycovs[:, 0, 0:-1] = yc[:, 1::]
            else:
                ycovs[:, 0, :] = yc
            period = self.period
            while last_step + future - (period - 1) > last_step + 1:
                period += self.period
            # this seems like we are looking ahead, but it will not use the last coordinate,
            # which is the only new point added
            ycovs[:, 1, period - 1::] = self.Ymat[
                :, last_step - rg: last_step + future - (period - 1)]

        Y = self.Yseq.predict_future(
            data_in=self.Ymat[ind, last_step - rg: last_step],
            covariates=covs,
            ycovs=ycovs,
            future=future,
            bsize=bsize,
            normalize=False,
        )

        if normalize:
            Y = Y - self.mini
            Y = Y * self.s[ind, None] + self.m[ind, None]
            return Y
        else:
            return Y

    def rolling_validation(self, Ymat, tau=24, n=7, bsize=90, alpha=0.3):
        prevX = self.X.clone()
        prev_index = self.end_index
        out = self.predict(
            last_step=self.end_index,
            future=tau,
            bsize=bsize,
            normalize=self.normalize,
        )
        out_global = self.predict_global(
            np.arange(self.Ymat.shape[0]),
            last_step=self.end_index,
            future=tau,
            normalize=self.normalize,
            bsize=bsize,
        )
        predicted_values = []
        actual_values = []
        predicted_values_global = []
        S = out[:, -tau::]
        S_g = out_global[:, -tau::]
        predicted_values += [S]
        predicted_values_global += [S_g]
        R = Ymat[:, self.end_index: self.end_index + tau]
        actual_values += [R]
        print("Current window wape:{}".format(wape(S, R)))

        self.Xseq = self.Xseq.eval()
        self.Yseq.seq = self.Yseq.seq.eval()

        for i in range(n - 1):
            Xn = self.recover_future_X(
                last_step=self.end_index + 1,
                future=tau,
                num_epochs=100000,
                alpha=alpha,
                vanilla=True
            )
            self.X = torch.cat([self.X, Xn], dim=1)
            self.end_index += tau
            out = self.predict(
                last_step=self.end_index,
                future=tau,
                bsize=bsize,
                normalize=self.normalize,
            )
            out_global = self.predict_global(
                np.arange(self.Ymat.shape[0]),
                last_step=self.end_index,
                future=tau,
                normalize=self.normalize,
                bsize=bsize,
            )
            S = out[:, -tau::]
            S_g = out_global[:, -tau::]
            predicted_values += [S]
            predicted_values_global += [S_g]
            R = Ymat[:, self.end_index: self.end_index + tau]
            actual_values += [R]
            print("Current window wape:{}".format(wape(S, R)))

        predicted = np.hstack(predicted_values)
        predicted_global = np.hstack(predicted_values_global)
        actual = np.hstack(actual_values)

        dic = {}
        dic["wape"] = wape(predicted, actual)
        dic["mape"] = mape(predicted, actual)
        dic["smape"] = smape(predicted, actual)
        dic["mae"] = np.abs(predicted - actual).mean()
        dic["rmse"] = np.sqrt(((predicted - actual) ** 2).mean())
        dic["nrmse"] = dic["rmse"] / np.sqrt(((actual) ** 2).mean())

        dic["wape_global"] = wape(predicted_global, actual)
        dic["mape_global"] = mape(predicted_global, actual)
        dic["smape_global"] = smape(predicted_global, actual)
        dic["mae_global"] = np.abs(predicted_global - actual).mean()
        dic["rmse_global"] = np.sqrt(((predicted_global - actual) ** 2).mean())
        dic["nrmse_global"] = dic["rmse"] / np.sqrt(((actual) ** 2).mean())

        baseline = Ymat[:, Ymat.shape[1] - n * tau - tau: Ymat.shape[1] - tau]
        dic["baseline_wape"] = wape(baseline, actual)
        dic["baseline_mape"] = mape(baseline, actual)
        dic["baseline_smape"] = smape(baseline, actual)
        self.X = prevX
        self.end_index = prev_index

        return dic
