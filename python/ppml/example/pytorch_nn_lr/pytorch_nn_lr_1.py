import numpy as np
import pandas as pd

import torch
from torch import nn
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.algorithms.psi import PSI


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

    def forward(self, x):
        x = torch.stack(x)
        x = torch.sum(x, dim=0) # above two act as interactive layer, CAddTable
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    df_train = pd.read_csv('./python/ppml/example/pytorch_nn_lr/data/diabetes-vfl-1.csv')    
    
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    df_train = df_train[df_train['ID'].isin(intersection)]

    df_x = df_train.drop('Outcome', 1)
    df_y = df_train['Outcome']
    
    x = df_x.to_numpy(dtype="float32")
    y = np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1)
    
    model = LocalModel(len(df_x.columns))
    loss_fn = nn.BCELoss()
    server_model = ServerModel()
    ppl = Estimator.from_torch(client_model=model,
                               client_id='1',
                               loss_fn=loss_fn,
                               optimizer_cls=torch.optim.SGD,
                               optimizer_args={'lr':1e-3},
                               target='localhost:8980',
                               server_model=server_model)
    response = ppl.fit(x, y)
    result = ppl.predict(x)
