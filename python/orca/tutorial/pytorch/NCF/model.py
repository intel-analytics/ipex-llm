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
# ==============================================================================
# Most of the pytorch code is adapted from guoyang9's NCF implementation for
# ml-1m dataset.
# https://github.com/guoyang9/NCF
#

import numpy as np
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None,
                 cat_feats_dim=0, numeric_feats_dim=0, num_embed_cat_feats=None):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights;
        cat_feats_dim: number of categorical features;
        numeric_feats_dim: number of numerical features;
        num_embed_cat_feats: list of num_embeddings for each categorical features.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.cat_feats_dim = cat_feats_dim
        self.numeric_feats_dim = numeric_feats_dim

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)
        self.embed_catFeats_MLP = [nn.Embedding(num_embed_cat_feats[i], factor_num)
                                   for i in range(cat_feats_dim)]

        input_size = factor_num * (2 + cat_feats_dim) + numeric_feats_dim
        output_size = factor_num * (2 ** (num_layers - 1))
        MLP_modules = []
        MLP_modules.append(nn.Dropout(p=self.dropout))
        MLP_modules.append(nn.Linear(input_size, output_size))
        MLP_modules.append(nn.ReLU())
        for i in range(num_layers-1):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(output_size, output_size//2))
            MLP_modules.append(nn.ReLU())
            output_size = output_size // 2
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        output_modules = []
        output_modules.append(nn.Linear(predict_size, 1))
        output_modules.append(nn.Sigmoid())
        self.predict_layer = nn.Sequential(*output_modules)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, *args):
        user, item = args[0], args[1]
        user = user.type(torch.LongTensor)
        item = item.type(torch.LongTensor)
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            if self.cat_feats_dim > 0:
                for i in range(self.cat_feats_dim):
                    catFeat = args[2+i].type(torch.LongTensor)
                    embed_catFeats_MLP = self.embed_catFeats_MLP[i](catFeat)
                    interaction = torch.cat((interaction, embed_catFeats_MLP), -1)
            if self.numeric_feats_dim > 0:
                numeric_feats = torch.stack(args[2+self.cat_feats_dim:], dim=1) \
                    .type(torch.FloatTensor)
                interaction = torch.cat((interaction, numeric_feats), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
