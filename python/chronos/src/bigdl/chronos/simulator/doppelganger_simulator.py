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

import pickle
import os
import numpy as np

from bigdl.chronos.simulator.doppelganger.util import gen_attribute_input_noise,\
    gen_feature_input_noise, gen_feature_input_data_free, renormalize_per_sample
from bigdl.chronos.simulator.doppelganger.output import OutputType

import torch
import torch.nn.functional as F

MODEL_PATH = "doppelganger.ckpt"
FEATURE_OUTPUT = "feature.output.ckpt"
ATTRIBUTE_OUTPUT = "attribute.output.ckpt"


class DPGANSimulator:
    '''
    Doppelganger Simulator for time series generation.
    The codes and algorithm are adapted from https://github.com/fjxmlzn/DoppelGANger.
    '''
    def __init__(self,
                 L_max,
                 sample_len,
                 feature_dim,
                 num_real_attribute,
                 discriminator_num_layers=5,
                 discriminator_num_units=200,
                 attr_discriminator_num_layers=5,
                 attr_discriminator_num_units=200,
                 attribute_num_units=100,
                 attribute_num_layers=3,
                 feature_num_units=100,
                 feature_num_layers=1,
                 attribute_input_noise_dim=5,
                 addi_attribute_input_noise_dim=5,
                 d_gp_coe=10,
                 attr_d_gp_coe=10,
                 g_attr_d_coe=1,
                 d_lr=0.001,
                 attr_d_lr=0.001,
                 g_lr=0.001,
                 g_rounds=1,
                 d_rounds=1,
                 seed=0,
                 num_threads=None,
                 ckpt_dir=".",
                 checkpoint_every_n_epoch=0):
        '''
        Initialize a doppelganger simulator.

        :param L_max: the maximum length of your feature.
        :param sample_len: the sample length to control LSTM length, should be a divider to L_max
        :param feature_dim: dimention of the feature
        :param num_real_attribute: the length of you attribute, which should be equal to the
               len(data_attribute).
        :param discriminator_num_layers: MLP layer num for discriminator.
        :param discriminator_num_units: MLP hidden unit for discriminator.
        :param attr_discriminator_num_layers: MLP layer num for attr discriminator.
        :param attr_discriminator_num_units: MLP hidden unit for attr discriminator.
        :param attribute_num_units: MLP layer num for attr generator/addi attr generator.
        :param attribute_num_layers:  MLP hidden unit for attr generator/addi attr generator.
        :param feature_num_units: LSTM hidden unit for feature generator.
        :param feature_num_layers: LSTM layer num for feature generator.
        :param attribute_input_noise_dim: noise data dim for attr generator.
        :param addi_attribute_input_noise_dim: noise data dim for addi attr generator.
        :param d_gp_coe: gradient penalty ratio for d loss.
        :param attr_d_gp_coe: gradient penalty ratio for attr d loss.
        :param g_attr_d_coe: ratio between feature loss and attr loss for g loss.
        :param d_lr: learning rate for discriminator.
        :param attr_d_lr: learning rate for attr discriminator.
        :param g_lr: learning rate for genereators.
        :param g_rounds: g rounds.
        :param d_rounds: d rounds.
        :param seed: random seed.
        :param num_threads: num of threads to be used for training.
        :param ckpt_dir: The checkpoint location, defaults to the working dir.
        :param checkpoint_every_n_epoch: checkpoint every n epoch, defaults to 0
               for no checkpoints.
        '''
        # additional settings
        from pytorch_lightning import seed_everything
        seed_everything(seed=seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir_model = os.path.join(self.ckpt_dir, "model")
        self.checkpoint_every_n_epoch = checkpoint_every_n_epoch
        self.sample_len = sample_len
        self.L_max = L_max
        self.feature_dim = feature_dim
        self.num_real_attribute = num_real_attribute

        # hparam saving
        self.params = {"discriminator_num_layers": discriminator_num_layers,
                       "discriminator_num_units": discriminator_num_units,
                       "attr_discriminator_num_layers": attr_discriminator_num_layers,
                       "attr_discriminator_num_units": attr_discriminator_num_units,
                       "attribute_num_units": attribute_num_units,
                       "attribute_num_layers": attribute_num_layers,
                       "feature_num_units": feature_num_units,
                       "feature_num_layers": feature_num_layers,
                       "attribute_input_noise_dim": attribute_input_noise_dim,
                       "addi_attribute_input_noise_dim": addi_attribute_input_noise_dim,
                       "d_gp_coe": d_gp_coe,
                       "attr_d_gp_coe": attr_d_gp_coe,
                       "g_attr_d_coe": g_attr_d_coe,
                       "d_lr": d_lr,
                       "attr_d_lr": attr_d_lr,
                       "g_lr": g_lr,
                       "g_rounds": g_rounds,
                       "d_rounds": d_rounds}

        # model init
        self.model = None  # model will be lazy built since the dim will depend on the data

    def fit(self,
            data_feature,
            data_attribute,
            data_gen_flag,
            feature_outputs,
            attribute_outputs,
            epoch=1,
            batch_size=32):
        '''
        Fit on the training data(typically the private data).

        :param data_feature: Training features, in numpy float32 array format.
               The size is [(number of training samples) x (maximum length)
               x (total dimension of features)]. Categorical features are stored
               by one-hot encoding; for example, if a categorical feature has 3
               possibilities, then it can take values between [1., 0., 0.],
               [0., 1., 0.], and [0., 0., 1.]. Each continuous feature should be
               normalized to [0, 1] or [-1, 1]. The array is padded by zeros after
               the time series ends.
        :param data_attribute: Training attributes, in numpy float32 array format. The size is
               [(number of training samples) x (total dimension of attributes)]. Categorical
               attributes are stored by one-hot encoding; for example, if a categorical
               attribute has 3 possibilities, then it can take values between [1., 0., 0.],
               [0., 1., 0.], and [0., 0., 1.]. Each continuous attribute should be normalized
               to [0, 1] or [-1, 1].
        :param data_gen_flag: Flags indicating the activation of features, in numpy float32
               array format. The size is [(number of training samples) x (maximum length)].
               1 means the time series is activated at this time step, 0 means the time series
               is inactivated at this timestep.
        :param feature_outputs: A list of Output indicates the meta data of data_feature.
        :param attribute_outputs: A list of Output indicates the meta data of data_attribute.
        :param epoch: training epoch.
        :param batch_size: training batchsize.
        '''
        # data preparation
        real_data = {}
        real_data["data_feature"] = data_feature
        real_data["data_attribute"] = data_attribute
        real_data["data_gen_flag"] = data_gen_flag
        from bigdl.chronos.simulator.doppelganger.data_module import DoppelGANgerDataModule
        self.data_module = DoppelGANgerDataModule(real_data=real_data,
                                                  feature_outputs=feature_outputs,
                                                  attribute_outputs=attribute_outputs,
                                                  sample_len=self.sample_len,
                                                  batch_size=batch_size)

        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(dirpath=self.ckpt_dir_model,
                                              save_top_k=-1,
                                              every_n_epochs=self.checkpoint_every_n_epoch)
        if self.checkpoint_every_n_epoch != 0:
            with open(os.path.join(self.ckpt_dir, FEATURE_OUTPUT), "wb") as f:
                pickle.dump(self.data_module.data_feature_outputs, f)
            with open(os.path.join(self.ckpt_dir, ATTRIBUTE_OUTPUT), "wb") as f:
                pickle.dump(self.data_module.data_attribute_outputs, f)

        # build the model
        from bigdl.chronos.simulator.doppelganger.doppelganger_pl import DoppelGANger_pl
        self.model = DoppelGANger_pl(data_feature_outputs=self.data_module.data_feature_outputs,
                                     data_attribute_outputs=self.data_module.data_attribute_outputs,
                                     L_max=self.L_max,
                                     sample_len=self.sample_len,
                                     num_real_attribute=self.num_real_attribute,
                                     **self.params)
        from pytorch_lightning import Trainer
        self.trainer = Trainer(logger=False,
                               callbacks=checkpoint_callback,
                               max_epochs=epoch,
                               default_root_dir=self.ckpt_dir)

        # fit!
        self.trainer.fit(self.model, self.data_module)

    def generate(self, sample_num=1, batch_size=32):
        '''
        Generate synthetic data with similar distribution as training data.

        :param sample_num: How many samples to be generated.
        :param batch_size: batch size to generate.
        '''
        # set to inference mode
        self.model.eval()
        total_generate_num_sample = sample_num

        # generate noise and inputs
        real_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        addi_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        feature_input_noise = gen_feature_input_noise(total_generate_num_sample, self.model.length)
        feature_input_data = gen_feature_input_data_free(total_generate_num_sample,
                                                         self.model.sample_len,
                                                         self.feature_dim)
        real_attribute_input_noise = torch.from_numpy(real_attribute_input_noise).float()
        addi_attribute_input_noise = torch.from_numpy(addi_attribute_input_noise).float()
        feature_input_noise = torch.from_numpy(feature_input_noise).float()
        feature_input_data = torch.from_numpy(feature_input_data).float()

        # generate
        features, attributes, gen_flags, lengths\
            = self.model.sample_from(real_attribute_input_noise,
                                     addi_attribute_input_noise,
                                     feature_input_noise,
                                     feature_input_data,
                                     batch_size=batch_size)

        # renormalize (max, min)
        features, attributes = renormalize_per_sample(
            features, attributes, self.model.data_feature_outputs,
            self.model.data_attribute_outputs, gen_flags,
            num_real_attribute=self.num_real_attribute)  # -2 for addi attr

        # post-process the attributes
        output_list = []
        current_idx = 0
        for i in range(len(self.model.data_attribute_outputs)):
            output_it = self.model.data_attribute_outputs[i]
            if output_it.type_ == OutputType.DISCRETE:
                sub_output = F.softmax(torch.from_numpy(attributes[:, current_idx:
                                                                   current_idx+output_it.dim]))
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1),
                                                num_classes=output_it.dim)
                output_list.append(sub_output_discrete)
            current_idx += output_it.dim
        attributes = torch.cat(output_list, dim=1).numpy()

        return features, attributes, gen_flags, lengths

    def save(self, path_dir):
        '''
        Save the simulator.

        :param path_dir: saving path
        '''
        path_dir_model = os.path.join(path_dir, "model")
        self.trainer.save_checkpoint(os.path.join(path_dir_model, MODEL_PATH))
        with open(os.path.join(path_dir, FEATURE_OUTPUT), "wb") as f:
            pickle.dump(self.data_module.data_feature_outputs, f)
        with open(os.path.join(path_dir, ATTRIBUTE_OUTPUT), "wb") as f:
            pickle.dump(self.data_module.data_attribute_outputs, f)

    def load(self,
             path_dir,
             model_version=MODEL_PATH):
        '''
        Load the simulator.

        :param path_dir: saving path
        :param model_version: model version(filename) you would like to load.
        '''
        with open(os.path.join(path_dir, FEATURE_OUTPUT), "rb") as f:
            data_feature_outputs = pickle.load(f)
        with open(os.path.join(path_dir, ATTRIBUTE_OUTPUT), "rb") as f:
            data_attribute_outputs = pickle.load(f)
        path_dir_model = os.path.join(path_dir, "model")
        from bigdl.chronos.simulator.doppelganger.doppelganger_pl import DoppelGANger_pl
        self.model =\
            DoppelGANger_pl.load_from_checkpoint(os.path.join(path_dir_model, model_version),
                                                 data_feature_outputs=data_feature_outputs,
                                                 data_attribute_outputs=data_attribute_outputs)
