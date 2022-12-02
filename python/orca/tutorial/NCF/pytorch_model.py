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
# Some of the code is adapted from
# https://github.com/guoyang9/NCF/blob/master/model.py
#

import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None,
                 sparse_feats_input_dims=None,
                 sparse_feats_embed_dims=8,
                 num_dense_feats=0):
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
        sparse_feats_input_dims: the list of input dimensions of sparse features;
        sparse_feats_embed_dims: the list of embedding dimensions of sparse features;
        num_dense_feats: number of dense features.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.sparse_feats_input_dims = sparse_feats_input_dims \
            if sparse_feats_input_dims is not None else []
        self.num_dense_feats = num_dense_feats
        self.num_sparse_feats = len(self.sparse_feats_input_dims)
        self.sparse_feats_embed_dims = sparse_feats_embed_dims \
            if isinstance(sparse_feats_embed_dims, list) \
            else [sparse_feats_embed_dims] * self.num_sparse_feats

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num,
                                           factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num,
                                           factor_num * (2 ** (num_layers - 1)))
        self.embed_catFeats_MLP = [nn.Embedding(self.sparse_feats_input_dims[i],
                                                self.sparse_feats_embed_dims[i])
                                   for i in range(self.num_sparse_feats)]

        input_size = factor_num * (2 ** num_layers) + \
            sum(self.sparse_feats_embed_dims) + num_dense_feats
        output_size = factor_num * (2 ** (num_layers - 1))
        MLP_modules = []
        for i in range(num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, output_size))
            MLP_modules.append(nn.ReLU())
            input_size = output_size
            output_size = output_size // 2

        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            for embed_MLP in self.embed_catFeats_MLP:
                nn.init.normal_(embed_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)
            for i in range(len(self.embed_catFeats_MLP)):
                self.embed_catFeats_MLP[i].weight.data.copy_(
                    self.MLP_model.embed_catFeats_MLP[i].weight)

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

    def forward(self, user, item, *args):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            for i in range(self.num_sparse_feats):
                embed_catFeats_MLP = self.embed_catFeats_MLP[i](args[i])
                interaction = torch.cat((interaction, embed_catFeats_MLP), -1)
            for i in range(self.num_dense_feats):
                interaction = torch.cat((interaction, args[i+self.num_sparse_feats]), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
