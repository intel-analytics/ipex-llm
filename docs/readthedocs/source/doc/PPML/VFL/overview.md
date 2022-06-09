# Vertical Federated Learning
Vertical Federated Learning (VFL) is a federated machine learning case where multiple data sets share the same sample ID space but differ in feature space. 

VFL is supported in BigDL PPML. It allows users to train a federated machine learning model where data features are held by different parties. In BigDL PPML, the following VFL scenarios are supported.
* **Private Set Intersection**: To get data intersection of different VFL parties.
* **Neural Network Model**: To train common neural network model with Pytorch or Tensorflow backend across VFL parties.
* **FGBoost Model**: To train gradient boosted decision tree (GBDT) model across multiple VFL parties.

For each scenario, some quick starts are available in above links.

## Key Concepts
A **FL Server** is a gRPC server to handle requests from FL Client. A **FL Client** is a gRPC client to send requests to FL Server. These requests include:
* serialized model to use in training at FL Server
* some model related instance, e.g. loss function, optimizer
* the Tensor which FL Server and FL Client interact, e.g. predict output, label, gradient

A **FL Context** is a singleton holding a FL Client instance. By default, only one instance is held in a FL application. And the gRPC channel in this singleton instance could be reused in multiple algorithms.


## Lifecycle

## Fault Tolerance
