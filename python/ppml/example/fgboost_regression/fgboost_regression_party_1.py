#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd
import click

from bigdl.ppml.fl.algorithms.psi import PSI

# the preprocess code is mainly from
# https://www.kaggle.com/code/pablocastilla/predict-house-prices-with-xgboost-regression/notebook
def preprocess(train_dataset, test_dataset):
    # takes Pandas DataFrame of raw data and output the preprocessed data
    # raw data may have any type of data, preprocessed data only have numerical data
    categorical_features_all= \
    ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
     'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
     'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
     'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
     'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
     'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
     'MiscFeature','SaleType','SaleCondition']
    categorical_features_party = list(set(train_dataset.columns) & set(categorical_features_all))

    every_column_non_categorical= [col for col in train_dataset.columns \
        if col not in categorical_features_party and col not in ['Id'] ]
    # log transform skewed numeric features:
    numeric_feats = train_dataset[every_column_non_categorical] \
        .dtypes[train_dataset.dtypes != "object"].index
    train_dataset[numeric_feats] = np.log1p(train_dataset[numeric_feats])

    every_column_non_categorical= [col for col in test_dataset.columns \
        if col not in categorical_features_party and col not in ['Id'] ]
    numeric_feats = test_dataset[every_column_non_categorical] \
        .dtypes[test_dataset.dtypes != "object"].index
    test_dataset[numeric_feats] = np.log1p(test_dataset[numeric_feats])
    # All features with NaN values in dataset, some of them do not exist after split into parties
    # Thus we need to get intersections, to get the features in particular party
    categorical_features_with_nan_all = \
    ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
     'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']
    categorical_features_with_nan_party = list(set(train_dataset.columns) & set(categorical_features_with_nan_all))

    numeric_features_with_nan_all = ['LotFrontage', 'GarageYrBlt']
    numeric_features_with_nan_party = list(set(train_dataset.columns) & set(numeric_features_with_nan_all))

    def ConvertNaNToNAString(data, columnList):
        for x in columnList:
            data[x] = str(data[x])

    def FillNaWithMean(data, columnList):
        for x in columnList:
            data[x] = data[x].fillna(data[x].mean())

    ConvertNaNToNAString(train_dataset, categorical_features_with_nan_party)
    ConvertNaNToNAString(test_dataset, categorical_features_with_nan_party)
    FillNaWithMean(train_dataset, numeric_features_with_nan_party)
    FillNaWithMean(test_dataset, numeric_features_with_nan_party)

    train_dataset = pd.get_dummies(train_dataset, columns=categorical_features_party)
    test_dataset = pd.get_dummies(test_dataset, columns=categorical_features_party)
    every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]
    y = train_dataset[['SalePrice']] if 'SalePrice' in train_dataset else None
    return train_dataset[every_column_except_y], y, test_dataset


@click.command()
@click.option('--load_model', default=False)
def run_client(load_model):
    client_id = 1
    init_fl_context(client_id)

    df_train = pd.read_csv('./python/ppml/example/fgboost_regression/data/house-prices-train-1.csv')
    df_train['Id'] = df_train['Id'].astype(str)

    df_test = pd.read_csv('./python/ppml/example/fgboost_regression/data/house-prices-test-1.csv')
    df_test['Id'] = df_test['Id'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['Id']))
    df_train = df_train[df_train['Id'].isin(intersection)]

    x, y, x_test = preprocess(df_train, df_test)

    if load_model:
        loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')
        loaded.fit(x, feature_columns=x.columns, num_round=10)
    else:
        fgboost_regression = FGBoostRegression()

        # party 1 does not own label, so directly pass all the features
        fgboost_regression.fit(x, feature_columns=x.columns, num_round=10)

        fgboost_regression.save_model('/tmp/fgboost_model_1.json')
        loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')

    result = loaded.predict(x_test, feature_columns=x_test.columns)

    # print first 5 results
    for i in range(5):
        print(f"{i}-th result of FGBoost predict: {result[i]}")

if __name__ == '__main__':
    run_client()