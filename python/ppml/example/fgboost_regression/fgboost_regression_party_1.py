from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
df_train = pd.read_csv('house-prices-train-1.csv')
fgboost_regression = FGBoostRegression()

# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)

df_test = pd.read_csv('house-prices-test-1')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)

fgboost_regression.save_model('/tmp/fgboost_model_1.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')
