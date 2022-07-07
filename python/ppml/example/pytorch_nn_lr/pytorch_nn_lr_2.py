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


if __name__ == '__main__':
    df_train = pd.read_csv('./python/ppml/example/pytorch_nn_lr/data/diabetes-vfl-2.csv')

    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    df_train = df_train[df_train['ID'].isin(intersection)]
    
    df_x = df_train
    x = df_x.to_numpy(dtype="float32")
    y = None
    
    model = LocalModel(len(df_x.columns))
    loss_fn = nn.BCELoss()
    ppl = Estimator.from_torch(client_model=model,
                               client_id='2',
                               loss_fn=loss_fn,
                               optimizer_cls=torch.optim.SGD,
                               optimizer_args={'lr':1e-3},
                               target='localhost:8980')
    response = ppl.fit(x, y)
    result = ppl.predict(x)
