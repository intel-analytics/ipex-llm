import os
import time
import argparse
import numpy as np
import pandas as pd 
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F 
from model import NCF

#Step 0: Parameters And Configuration

Config={  
    "dataset": "ml-1m",# dataset name    
    "model": "NeuMF-end",# model name 
    "main_path":"./NCF-Data/",# paths
    "model_path" :  './models/',  
    "out": True,# save model or not
    "cluster_mode": "local",
    "lr": 0.001,# learning rate
    "dropout": 0.0,# dropout rate
    "batch_size": 256,# batch size for training
    "epochs": 20,# training epoches
    "top_k": 10,# compute metrics@top_k
    "factor_num": 32,# predictive factors numbers in the model
    "num_layers": 3,# number of layers in MLP model
    "num_ng": 4,# sample negative items for training
    "test_num_ng": 0,# sample part of negative items for testing
    "backend": "ray", # backend used in estimator, "ray" or "spark" are supported
    "model_dir": "./model_dir/",
}

Config["train_rating"]=Config["main_path"]+ Config["dataset"]+".train.rating"
Config["test_rating"]=Config["main_path"]+ Config["dataset"]+".test.rating"
Config["test_negative"]=Config["main_path"]+ Config["dataset"]+".test.negative"
Config["GMF_model_path"]=Config["main_path"]+ 'GMF.pth'
Config["MLP_model_path"]=Config["main_path"]+ 'MLP.pth'
Config["NeuMF_model_path"]=Config["main_path"]+ 'NeuMF.pth'


#Step 1: Init Orca Context

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
init_orca_context(cores=1, memory="8g")

#Step 2: Define Train Dataset

from bigdl.orca.data import XShards
from bigdl.orca.data.pandas import read_csv

def preprocess_data():
    #calculate user_num and item_num 
    train_data = pd.read_csv(
        Config["train_rating"], 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    
    # load ratings as a dok matrix
    train_data = train_data.values.tolist()
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for x in train_data:
        train_mat[x[0], x[1]] = 1
        
    # load train_data_X as XShards
    train_data_X = read_csv(
        Config["train_rating"], 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})   
    
    #load test_data_X as XShards
    test_data_X = []
    with open(Config["test_negative"], 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data_X.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data_X.append([u, int(i)])
            line = fd.readline()
    test_data_X= pd.DataFrame(test_data_X,columns=['user', 'item'])
    test_data_X.to_csv(Config["main_path"]+ Config["dataset"]+".test_negative", index=False, header=0) 
    test_data_X = read_csv(Config["main_path"]+ Config["dataset"]+".test_negative", 
        sep=',', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    return train_data_X,test_data_X,user_num, item_num, train_mat

# prepare the train and test datasets
train_data_X, test_data_X, Config["user_num"], Config["item_num"], train_mat = preprocess_data()

# construct the train and test xshards
def transform_to_dict(data,is_training=False):
    data_X=data.values.tolist()
    
    # need sampling when training
    if(is_training==True):
        features_ps=data_X
        
        features_ng = []
        for x in features_ps:
            u = x[0]
            for t in range(Config["num_ng"]):
                j = np.random.randint(Config["item_num"])
                while (u, j) in train_mat:
                    j = np.random.randint(Config["item_num"])
                features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(features_ps))]
        labels_ng = [0 for _ in range(len(features_ng))]

        features_fill = features_ps + features_ng
        labels_fill = labels_ps + labels_ng      
        return {"x": np.array(features_fill,np.int64), "y": np.array(labels_fill,np.float)}  
    # testing          
    return {"x":np.array(data_X,np.int64), "y": np.array([0 for _ in range(len(data_X))],np.float)}

train_shards = train_data_X.transform_shard(transform_to_dict,True)#is_training=True
test_shards = test_data_X.transform_shard(transform_to_dict,False)

#Step 3: Define the Model

# create the model
def model_creator(config):
    model = NCF(Config["user_num"], Config["item_num"],Config["factor_num"], Config["num_layers"], Config["dropout"], Config["model"]) # a torch.nn.Module
    model.train()
    return model

#create the optimizer
def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr= Config["lr"])

#define the loss function
loss_function = nn.BCEWithLogitsLoss()

#Step 4: Fit with Orca Estimator

from bigdl.orca.learn.pytorch import Estimator 
from bigdl.orca.learn.metrics import Accuracy

# create the estimator
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,loss=loss_function, metrics=[Accuracy()],backend=Config["backend"])# backend="ray" or "spark"

# fit the estimator
est.fit(data=train_shards, epochs=1,batch_size=Config["batch_size"],feature_cols=["x"],label_cols =["y"])

#Step 5: Save and Load the Model

# save the model
est.save("NCF_model")

# load the model
est.load("NCF_model")

# evaluate the model
result = est.evaluate(data=test_shards)
for r in result:
    print(r, ":", result[r])

# stop orca context when program finishes
stop_orca_context()



