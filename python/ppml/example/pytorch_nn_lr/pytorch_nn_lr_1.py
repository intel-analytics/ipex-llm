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

from typing import List
import numpy as np
import pandas as pd
import click

import torch
from torch import Tensor, nn
from bigdl.ppml.fl.estimator import Estimator


class LocalModel(nn.Module):
    def __init__(self, num_feature) -> None:
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        x = self.dense(x)
        return x


class ServerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: List[Tensor]):
        x = torch.stack(x)
        x = torch.sum(x, dim=0) # above two act as interactive layer, CAddTable
        x = self.sigmoid(x)
        return x



@click.command()
@click.option('--load_model', default=False)
def run_client(load_model):
    # fl_server = FLServer(2)
    # fl_server.build()
    # fl_server.start()
    df_train = pd.read_csv('./python/ppml/example/pytorch_nn_lr/data/diabetes-vfl-1.csv')    
    
    # this should wait for the merge of 2 FLServer (Py4J Java gRPC and Python gRPC)
    # df_train['ID'] = df_train['ID'].astype(str)
    # psi = PSI()
    # intersection = psi.get_intersection(list(df_train['ID']))
    # df_train = df_train[df_train['ID'].isin(intersection)]

    df_x = df_train.drop('Outcome', 1)
    df_y = df_train['Outcome']
    
    x = df_x.to_numpy(dtype="float32")
    y = np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1)
    
    loss_fn = nn.BCELoss()

    if load_model:
        model = torch.load('/tmp/pytorch_client_model_1.pt')
        ppl = Estimator.from_torch(client_model=model,
                                   client_id='1',
                                   loss_fn=loss_fn,
                                   optimizer_cls=torch.optim.SGD,
                                   optimizer_args={'lr':1e-5},
                                   target='localhost:8980')
        response = ppl.fit(x, y, 5)
    else:
        model = LocalModel(len(df_x.columns))
        
        server_model = ServerModel()
        ppl = Estimator.from_torch(client_model=model,
                                   client_id='1',
                                   loss_fn=loss_fn,
                                   optimizer_cls=torch.optim.SGD,
                                   optimizer_args={'lr':1e-5},
                                   target='localhost:8980',
                                   server_model=server_model)
        response = ppl.fit(x, y, 5)
        torch.save(ppl.model, '/tmp/pytorch_client_model_1.pt')
    result = ppl.predict(x)
    print(result[:5])

if __name__ == '__main__':
    run_client()