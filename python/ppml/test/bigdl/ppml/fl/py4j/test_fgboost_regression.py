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


from multiprocessing import Process
import unittest
from uuid import uuid4
import numpy as np
import pandas as pd
import os
import math

from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
from bigdl.ppml.fl.fl_server import FLServer
from bigdl.ppml.fl.utils import init_fl_context
from bigdl.ppml.fl.utils import FLTest

resource_path = os.path.join(os.path.dirname(__file__), "../resources")

def mock_process(client_id, data_train, data_test, target="localhost:8980"):
    # FLContext is a singleton in JVM, so another process initialization is needed
    # prepare_env()
    init_fl_context(client_id, target)

    df_train = pd.read_csv(os.path.join(resource_path, data_train))

    fgboost_regression = FGBoostRegression()
    if 'SalePrice' in df_train:
        df_x = df_train.drop('SalePrice', 1)
        df_y = df_train.filter(items=['SalePrice'])
        fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=15)
    else:
        fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=15)

    df_test = pd.read_csv(os.path.join(resource_path, data_test))
    result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)


class TestFGBoostRegression(FLTest):    
    xgboost_result = pd.read_csv(os.path.join(
            resource_path, "house-price-xgboost-submission.csv"))
    xgboost_result = xgboost_result['SalePrice'].to_numpy()
    @classmethod
    def setUpClass(cls) -> None:
        multiprocessing.set_start_method('spawn') 

    def setUp(self) -> None:
        self.update_available_port()
        self.fl_server = FLServer()
        self.fl_server.set_port(self.port)
        init_fl_context(1, self.target)
        # this explicit set is needed, default value is 'fork' on Unix
        # if 'fork', the resources would be inherited and thread crash would occur
        # (to be verified)
            

    def tearDown(self) -> None:
        self.fl_server.stop()

    def test_dummy_data(self):
        self.fl_server.set_client_num(1)
        self.fl_server.build()
        self.fl_server.start()
        x, y = np.ones([2, 3]), np.ones([2])
        fgboost_regression = FGBoostRegression()
        fgboost_regression.fit(x, y)
        result = fgboost_regression.predict(x)
        result

    def test_save_load(self):
        self.fl_server.set_client_num(1)
        self.fl_server.build()
        self.fl_server.start()
        df_train = pd.read_csv(
            os.path.join(resource_path, "house-prices-train-preprocessed.csv"))
        server_model_path = "/tmp/fgboost_server_model"
        fgboost_regression = FGBoostRegression(server_model_path=server_model_path)
        
        df_x = df_train.drop('SalePrice', 1)
        df_y = df_train.filter(items=['SalePrice'])
        
        fgboost_regression.fit(df_x, df_y,
            feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=1)
        tmp_file_name = f'/tmp/{str(uuid4())}'
        
        fgboost_regression.save_model(tmp_file_name)
        model_loaded = FGBoostRegression.load_model(tmp_file_name)
        
        df_test = pd.read_csv(os.path.join(resource_path, "house-prices-test-preprocessed.csv"))
        result = model_loaded.predict(df_test, feature_columns=df_x.columns)
        result = list(map(lambda x: math.exp(x), result))
        os.remove(server_model_path)
        os.remove(tmp_file_name)
        result

    def test_three_party(self):
        self.fl_server.set_client_num(3)
        self.fl_server.build()
        self.fl_server.start()
        mock_party1 = Process(target=mock_process, 
        args=(2, 'house-prices-train-preprocessed-1.csv', 'house-prices-test-preprocessed-1.csv', self.target))
        mock_party1.start()
        mock_party2 = Process(target=mock_process, 
        args=(3, 'house-prices-train-preprocessed-2.csv', 'house-prices-test-preprocessed-2.csv', self.target))
        mock_party2.start()        

        df_train = pd.read_csv(
            os.path.join(resource_path, "house-prices-train-preprocessed-0.csv"))
        fgboost_regression = FGBoostRegression()
        if 'SalePrice' in df_train:
            df_x = df_train.drop('SalePrice', 1)
            df_y = df_train.filter(items=['SalePrice'])
            
            fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=15)
        else:
            fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=15)
        
        df_test = pd.read_csv(os.path.join(resource_path, "house-prices-test-preprocessed-0.csv"))
        result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
        result = np.exp(result)
        assert len(result) == len(TestFGBoostRegression.xgboost_result)
        assert np.allclose(result, TestFGBoostRegression.xgboost_result, rtol=10, atol=10)

if __name__ == '__main__':
    unittest.main()
