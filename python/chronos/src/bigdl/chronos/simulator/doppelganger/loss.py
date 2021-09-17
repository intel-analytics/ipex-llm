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


import torch
import torch.autograd as autograd


EPS = 1e-8


def doppelganger_loss(d_fake,
                      attr_d_fake,
                      d_real,
                      attr_d_real,
                      g_attr_d_coe=1,
                      gradient_penalty=False,
                      discriminator=None,
                      attr_discriminator=None,
                      g_output_feature_train_tf=None,
                      g_output_attribute_train_tf=None,
                      real_feature_pl=None,
                      real_attribute_pl=None,
                      d_gp_coe=0,
                      attr_d_gp_coe=0,
                      ):
    batch_size = d_fake.shape[0]

    g_loss_d = -torch.mean(d_fake)
    g_loss_attr_d = -torch.mean(attr_d_fake)
    g_loss = g_loss_d + g_attr_d_coe * g_loss_attr_d

    if gradient_penalty:
        alpha_dim2 = torch.rand(batch_size, 1)
        alpha_dim3 = torch.unsqueeze(alpha_dim2, dim=2)
        differences_input_feature = g_output_feature_train_tf - real_feature_pl
        interpolates_input_feature = real_feature_pl + alpha_dim3 * differences_input_feature
        differences_input_attribute = g_output_attribute_train_tf - real_attribute_pl
        interpolates_input_attribute = real_attribute_pl + alpha_dim2 * differences_input_attribute

        interpolates_input_feature.requires_grad = True
        interpolates_input_attribute.requires_grad = True
        disc_interpolates = discriminator(interpolates_input_feature, interpolates_input_attribute)

        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=[interpolates_input_feature, interpolates_input_attribute],
                                  grad_outputs=torch.ones(disc_interpolates.shape),
                                  create_graph=True,
                                  retain_graph=True,
                                  allow_unused=True)
        slopes1 = torch.sum(torch.square(gradients[0]), dim=[1, 2])
        slopes2 = torch.sum(torch.square(gradients[1]), dim=1)
        slopes = torch.sqrt(slopes1 + slopes2 + EPS)
        d_loss_gp = ((slopes - 1) ** 2).mean()
    else:
        d_loss_gp = 0

    d_loss_fake = torch.mean(d_fake)
    d_loss_real = -torch.mean(d_real)
    d_loss = d_loss_fake + d_loss_real + d_gp_coe*d_loss_gp

    if gradient_penalty:
        alpha_dim2_attr = torch.rand(batch_size, 1)
        differences_input_attribute = g_output_attribute_train_tf - real_attribute_pl
        interpolates_input_attribute = real_attribute_pl +\
            alpha_dim2_attr * differences_input_attribute
        interpolates_input_attribute.requires_grad = True
        attr_disc_interpolates = attr_discriminator(interpolates_input_attribute)
        gradients_attr = autograd.grad(outputs=attr_disc_interpolates,
                                       inputs=interpolates_input_attribute,
                                       grad_outputs=torch.ones(attr_disc_interpolates.shape),
                                       create_graph=True,
                                       retain_graph=True,
                                       allow_unused=True)
        slopes1_attr = torch.sum(torch.square(gradients_attr[0]), dim=1)
        slopes_attr = torch.sqrt(slopes1_attr + EPS)
        attr_d_loss_gp = ((slopes_attr - 1) ** 2).mean()
    else:
        attr_d_loss_gp = 0

    attr_d_loss_fake = torch.mean(attr_d_fake)
    attr_d_loss_real = -torch.mean(attr_d_real)
    attr_d_loss = attr_d_loss_fake + attr_d_loss_real + attr_d_gp_coe * attr_d_loss_gp

    return g_loss, d_loss, attr_d_loss
