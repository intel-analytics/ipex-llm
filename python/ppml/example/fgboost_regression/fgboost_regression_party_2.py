from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
df_train = pd.read_csv('house-prices-train-2.csv')

# party 2 owns label, so split features and label first
df_x = df_train.drop('SalePrice', 1) # drop the label column
df_y = df_train.filter(items=['SalePrice']) # select the label column

fgboost_regression = FGBoostRegression()
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=100)

df_test = pd.read_csv('house-prices-test-2')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)

fgboost_regression.save_model('/tmp/fgboost_model_2.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_2.json')
