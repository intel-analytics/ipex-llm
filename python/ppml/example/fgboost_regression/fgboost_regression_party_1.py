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

from bigdl.ppml.fl.algorithms.psi import PSI

# the preprocess code is mainly from 
# https://www.kaggle.com/code/pablocastilla/predict-house-prices-with-xgboost-regression/notebook
def preprocess(train_dataset):
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

    # All features with NaN values in dataset, some of them do not exist after split into parties
    # Thus we need to get intersections, to get the features in particular party
    features_with_nan_all = \
    ['Alley','MasVnrType','BsmtQual','BsmtQual','BsmtCond','BsmtCond','BsmtExposure',
     'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']
    features_with_nan_party = list(set(train_dataset.columns) & set(features_with_nan_all))
    
    def ConverNaNToNAString(data, columnList):
        for x in columnList:       
            data[x] =str(data[x])  

    ConverNaNToNAString(train_dataset, features_with_nan_party)        
    
    train_dataset = pd.get_dummies(train_dataset, columns=categorical_features_party)
    every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]
    y = train_dataset[['SalePrice']] if 'SalePrice' in train_dataset else None
    return train_dataset[every_column_except_y], y


init_fl_context()

df_train = pd.read_csv('./python/ppml/example/fgboost_regression/data/house-prices-train-1.csv')

psi = PSI()
intersection = psi.get_intersection(df_train['Id'])
df_train = df_train.ix(intersection)

x, y = preprocess(df_train)

fgboost_regression = FGBoostRegression()

# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(x, feature_columns=x.columns, num_round=15)


