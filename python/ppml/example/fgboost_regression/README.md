# FGBoost Regression Quick Start
This example provides a step-by-step tutorial of running a FGBoost Regression task with 2 parties.
## Data
This section describe the data and preprocess, you may skip if you want to directly run the application.

This example use [House Price Dataset]().

To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts and preprocess them in advance. The split is taken by select every other column (code at [split script]()) and preprocessing includes filling NA values and applying one-hot encoding to categorical features (code at [preprocess script]()).

After preprocessing, we got a data file in `data` folder, including `house-prices-train-1.csv` with half of features and no label, and a data file `house-prices-train-2.csv` with another half of features and label `SalePrice`. The and test data `house-prices-test-1.csv` and `house-prices-test-2.csv`.


## Start FLServer
FL Server is required before running any federated applications. Please check [Start FL Server]() section.

## Write Client Code
The code is available in projects, including [Client 1 code]() and [Client 2 code](). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.

### Import and Load Data
First, import the package and initilize FL Context.
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
```
Then, read the data and split it into feature and label if necessary,

Party 1:
```python
df_train = pd.read_csv('house-prices-train-1.csv')
```

Party 2:
```python
df_train = pd.read_csv('house-prices-train-2.csv')

# party 2 owns label, so split features and label first
df_x = df_train.drop('SalePrice', 1) # drop the label column
df_y = df_train.filter(items=['SalePrice']) # select the label column
```
### Create Model and Train
Create the FGBoost model with default parameters and train
Party 1:
```python
fgboost_regression = FGBoostRegression()
# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)
```
Party 2:
```
fgboost_regression = FGBoostRegression()
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=100)
```
### Predict
Using the trained model to federated predict

Party 1:
```python
df_test = pd.read_csv('house-prices-test-1')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```
Party 2:
```python
df_test = pd.read_csv('house-prices-test-2')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```
### Save/Load
Save the model and load it back

Party 1:
```python
fgboost_regression.save_model('/tmp/fgboost_model_1.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')
```
Party 2:
```python
fgboost_regression.save_model('/tmp/fgboost_model_2.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_2.json')
```

