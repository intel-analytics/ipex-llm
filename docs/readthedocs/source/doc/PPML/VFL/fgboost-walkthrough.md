# FGBoost Walkthrough
**Note:** We recommend to go through [VFL Key Concepts]() before this walkthrough.

FGBoost supports gradient boosted tree algorithm in Vertical Federated Learning (VFL).


## Key Concepts
A **tree** is a decision tree (DT) with a set of nodes, each node represents a split in DT.

An **algorithm instance** is an instance to run FGBoost algorithm, each instance holds a list of DT for running boosted tree. The algorithm supports `FGBoostRegression` for regression task and `FGBoostClassification` for classification task.


## Example
This section provides an example of running a regression task with 2 parties.

Before running FGBoost algorithm, make sure FL Server is started. See [Start FL Server]()

This example use [House Price Dataset]().

To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts and preprocess them in advance. The split is taken by select every other column (code at [split script]()) and preprocessing includes filling NA values and applying one-hot encoding to categorical features (code at [preprocess script]()).

Now we got a data file `house-prices-train-1.csv` with half of features and no label, and a data file `house-prices-train-2.csv` with another half of features and label `SalePrice`.

### Train
Once dataset and FL Server is ready, we can start run client code to simulate the training of different parties. 

We will create an algorithm instance `FGBoostRegression` for each party, and read the corresponded data to fit the trees.

For party 1, start following code
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression

init_fl_context()
df_train = pd.read_csv('house-prices-train-1.csv')
fgboost_regression = FGBoostRegression()

# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)
```
For party 2, start following code
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression

init_fl_context()
df_train = pd.read_csv('house-prices-train-2.csv')
fgboost_regression = FGBoostRegression()

# party 2 owns label, so split features and label first
df_x = df_train.drop('SalePrice', 1) # drop the label column
df_y = df_train.filter(items=['SalePrice']) # select the label column
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=100)
```
For each `num_round` in `fit` method, a tree would be created. Note that all the parties should pass the same value of `num_round` otherwise the party with larger `num_round` would get block since it is continuously waiting for other parties whose `fit` processes are already done.
### Predict
So far, we have completed the training process of FGBoost. We could do federated predict from parties.

For party 1
```python
df_test = pd.read_csv('house-prices-test-1')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```
For party 2
```python
df_test = pd.read_csv('house-prices-test-2')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```

### Save/Load
We could save the algorithm instance to file in JSON format by
```python
fgboost_regression.save_model('fgboost.json')
```
and load model from file by
```python
fgboost_regression.load_model('fgboost.json')
```
