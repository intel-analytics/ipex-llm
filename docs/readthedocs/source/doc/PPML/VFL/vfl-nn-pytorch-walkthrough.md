# VFL NN Pytorch Walkthrough
**Note:** We recommend to go through [VFL Key Concepts]() before this walkthrough.

VFL NN Pytorch supports running arbitrary Pytorch model in Vertical Federated Learning (VFL).


## Key Concepts
A **client model** is an user-defined Pytorch model which runs local training and interacts with server model.

A **server model** is an user-defined Pytorch model, which is defined in client FL application and uploaded to FL Server.


## Quick Start
We provide some quick start project of VFL NN, see following

// TODO: following will be moved to another page
This section provides an example of running a regression task with 2 parties.

Before running FGBoost algorithm, make sure FL Server is started. See [Start FL Server]()

This example use [Indian Diabetes Dataset]().

To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts and preprocess them in advance. The split is taken by select every other column (code at [split script]()).

Now we got a data file `diabetes-vfl-1.csv` with half of features and label `Outcome`, and a data file `diabetes-vfl-2.csv` with another half of features and no label.
