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

import logging
import numpy as np
import pandas as pd
import click

import torch
from torch import nn
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI

fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)

class LocalModel(nn.Module):
    def __init__(self, num_feature) -> None:
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        x = self.dense(x)
        return x


@click.command()
@click.option('--load_model', default=False)
@click.option('--data_path', default="./data/diabetes-vfl-2.csv")
def run_client(load_model, data_path):
    init_fl_context(2)
    df_train = pd.read_csv(data_path)
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    df_train = df_train[df_train['ID'].isin(intersection)]
    
    df_x = df_train
    x = df_x.to_numpy(dtype="float32")
    y = None    
    
    loss_fn = nn.BCELoss()

    if load_model:
        model = torch.load('/tmp/pytorch_client_model_2.pt')
        ppl = Estimator.from_torch(client_model=model,
                                   loss_fn=loss_fn,
                                   optimizer_cls=torch.optim.SGD,
                                   optimizer_args={'lr':1e-4},
                                   client_model_path='/tmp/pytorch_client_model_2.pt')
        response = ppl.fit(x, y, 5)
    else:
        model = LocalModel(len(df_x.columns))
        ppl = Estimator.from_torch(client_model=model,
                                   loss_fn=loss_fn,
                                   optimizer_cls=torch.optim.SGD,
                                   optimizer_args={'lr':1e-4},
                                   client_model_path='/tmp/pytorch_client_model_2.pt')
        response = ppl.fit(x, y, 5)
    result = ppl.predict(x)
    print(result[:5])

if __name__ == '__main__':
    run_client()
