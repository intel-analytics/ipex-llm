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

# The Clear BSD License

# Copyright (c) 2019 Carnegie Mellon University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted (subject to the limitations in the disclaimer below) provided that
# the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of Carnegie Mellon University nor the names of its contributors
#       may be used to endorse or promote products derived from this software without
#       specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import time
import torch
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from zoo.chronos.simulator import DPGANSimulator
from zoo.chronos.simulator.doppelganger.output import Output, OutputType, Normalization

EPS = 1e-8


def autocorr(X, Y):
    Xm = torch.mean(X, 1).unsqueeze(1)
    Ym = torch.mean(Y, 1).unsqueeze(1)
    r_num = torch.sum((X - Xm) * (Y - Ym), 1)
    r_den = torch.sqrt(torch.sum((X - Xm)**2, 1) * torch.sum((Y - Ym)**2, 1))
    r_num[r_num == 0] = EPS
    r_den[r_den == 0] = EPS
    r = r_num / r_den
    r[r > 1] = 0
    r[r < -1] = 0
    return r


def get_autocorr(feature):
    feature = torch.from_numpy(feature)
    feature_length = feature.shape[1]
    autocorr_vec = torch.Tensor(feature_length - 2)
    for j in range(1, feature_length - 1):
        autocorr_vec[j - 1] = torch.mean(autocorr(feature[:, :-j],
                                                  feature[:, j:]))
    return autocorr_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1,
                        help="The number of cpu cores you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--epoch', type=int, default=25,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument('--datadir', type=str,
                        help="Use local npz file by default.")
    parser.add_argument('--plot_figures', type=bool, default=True,
                        help="Plot Figure 1, 6, 19 in the http://arxiv.org/abs/1909.13403")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Training batch size.")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint",
                        help="The checkpoint root dir.")
    parser.add_argument('--checkpoint_every_n_epoch', type=int, default=0,
                        help="Checkpoint per n epoch.")
    args = parser.parse_args()

    # real data
    real_data = np.load(args.datadir)

    # feature outputs
    feature_outputs = [Output(type_=OutputType.CONTINUOUS, dim=1,
                       normalization=Normalization.MINUSONE_ONE)]

    # attribute outputs
    attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9),
                         Output(type_=OutputType.DISCRETE, dim=3),
                         Output(type_=OutputType.DISCRETE, dim=2)]

    # init a generator
    doppelganger = DPGANSimulator(L_max=550,
                                  sample_len=10,
                                  feature_dim=1,
                                  num_real_attribute=3,
                                  num_threads=args.cores,
                                  ckpt_dir=args.checkpoint_path,
                                  checkpoint_every_n_epoch=args.checkpoint_every_n_epoch)

    # fit
    doppelganger.fit(data_feature=real_data["data_feature"],
                     data_attribute=real_data["data_attribute"],
                     data_gen_flag=real_data["data_gen_flag"],
                     feature_outputs=feature_outputs,
                     attribute_outputs=attribute_outputs,
                     epoch=args.epoch,
                     batch_size=args.batch_size)

    # save
    print("saving to", args.checkpoint_path)
    doppelganger.save(args.checkpoint_path)
    print("saved to", args.checkpoint_path)

    # generate synthetic dataset
    print("generating data")
    features, attributes, gen_flags, lengths =\
        doppelganger.generate(sample_num=real_data["data_feature"].shape[0])
    print("data generated")

    # evaluation(plot figure 1, 6, 19 on http://arxiv.org/abs/1909.13403)
    if args.plot_figures:

        # evalation(figure 1)
        autocorr_ori = get_autocorr(real_data["data_feature"]).numpy()
        autocorr_gen = get_autocorr(features).numpy()
        plt.figure(figsize=(10, 5))
        plt.plot(autocorr_ori)
        plt.plot(autocorr_gen)
        plt.legend(["real", "doppelganger"])
        plt.savefig("figure_1.png", format="png")

        # evaluation(figure 6)
        plt.figure(figsize=(10, 5))
        a = plt.hist((np.max(real_data["data_feature"], axis=(1, 2))
                     + np.min(real_data["data_feature"], axis=(1, 2)))/2, bins=100, alpha=0.7)
        plt.hist((np.max(features, axis=(1, 2)) +
                 np.min(features, axis=(1, 2)))/2, bins=a[1], alpha=0.7)
        plt.legend(["real", "doppelganger"])
        plt.savefig("figure_6.png", format="png")

        # evaluation(figure 19)
        plt.figure(figsize=(10, 5))
        plt.bar(np.array(range(9)), np.sum(real_data["data_attribute"][:, 0:9],
                axis=0), width=0.5, alpha=0.7)
        plt.bar(range(9), np.sum(attributes[:, 0:9], axis=0), width=0.5, alpha=0.7)
        plt.legend(["real", "doppelganger"])
        plt.savefig("figure_19.png", format="png")
