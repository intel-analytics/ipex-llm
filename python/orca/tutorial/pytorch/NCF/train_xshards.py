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
    "backend": "spark", # backend used in estimator, "ray" or "spark" are supported
    "user_num": 6040,
    "item_num": 3952,
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

#Step 2: Define Dataset

from bigdl.orca.data import XShards
from bigdl.orca.data.pandas import read_csv
from sklearn.model_selection import train_test_split

def preprocess_data():   
    data_X = read_csv(
        Config["dataset"]+"/ratings.dat", 
        sep="::", header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})  
    data_X=data_X.partition_by("user",5)#num_partitions=5 
    return data_X

# prepare the train and test datasets
data_X = preprocess_data()

# construct the train and test xshards
def transform_to_dict(data):
    data["user"],data["item"]=data["user"]-1,data["item"]-1
    data_X=data.values.tolist()
    
    #calculate a dok matrix
    train_mat = sp.dok_matrix((Config["user_num"], Config["item_num"]), dtype=np.int64)
    for row in data_X:
        train_mat[row[0], row[1]] = 1

    #negative sampling
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
    data_XY=pd.DataFrame(data=features_fill,columns=["user","item"])
    data_XY["y"]=labels_fill

    #split training set and testing set
    train_data, test_data=train_test_split(data_XY, test_size=0.2, random_state=100)

    #transform dataset into dict
    train_data, test_data=train_data.to_numpy(), test_data.to_numpy()
    train_data, test_data={"x": train_data[:,:2].astype(np.int64), "y": train_data[:,2].astype(np.float)}, {"x": test_data[:,:2].astype(np.int64), "y": test_data[:,2].astype(np.float)}  
    return train_data,test_data

train_shards, test_shards = data_X.transform_shard(transform_to_dict).split()

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
from bigdl.orca.learn.metrics import Accuracy,AUC

# create the estimator
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,loss=loss_function, metrics=[Accuracy(),AUC()],backend=Config["backend"])# backend="ray" or "spark"

# fit the estimator
est.fit(data=train_shards, epochs=5,batch_size=Config["batch_size"],feature_cols=["x"],label_cols =["y"])

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



