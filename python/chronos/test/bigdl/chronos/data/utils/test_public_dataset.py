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
import os
import pandas as pd
import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.data.utils.public_dataset import PublicDataset


class TestPublicDataset(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_init_get_dataset(self):
        name = 'nyc_taxi'
        path = '~/.chronos/dataset/'
        public_data = PublicDataset(name, path, redownload=False, with_split=False)

        # illegle input.
        with pytest.raises(AssertionError):
            PublicDataset(name, path, redownload=False).get_public_data(chunk_size='1024')

    def test_get_nyc_taxi(self):
        name = 'nyc_taxi'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv"

            public_data = PublicDataset(name, path, redownload=False)
            public_data.df = pd.read_csv(file_url, parse_dates=['timestamp'])

            tsdata = public_data.get_tsdata(target_col='value', dt_col='timestamp')
            assert set(tsdata.df.columns) == {'id', 'timestamp', 'value'}
            assert tsdata.df.shape == (10320, 3)
            tsdata._check_basic_invariants()

    def test_get_network_traffic(self):
        name = 'network_traffic'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/network-traffic/data/data.csv"

            public_data = PublicDataset(name, path, redownload=False)
            public_data.df = pd.read_csv(file_url)
            public_data.df.StartTime = pd.to_datetime(public_data.df.StartTime)
            public_data.df.AvgRate = public_data.df.AvgRate.apply(lambda x: float(x[:-4])
                                                                  if x.endswith("Mbps")
                                                                  else float(x[:-4])*1000)

            tsdata = public_data.get_tsdata(target_col=['AvgRate', 'total'], dt_col='StartTime')
            assert tsdata.df.shape == (8760, 5)
            assert set(tsdata.df.columns) == {'StartTime', 'EndTime', 'AvgRate', 'total', 'id'}
            tsdata._check_basic_invariants()

    def test_get_fsi(self):
        name = 'fsi'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/chronos-aiops/m_1932.csv"

            public_data = PublicDataset(name, path, redownload=False, with_split=False)
            public_data.df = pd.read_csv(file_url, usecols=[1, 2, 3],
                                         names=['time_step', 'cpu_usage', 'mem_usage'])
            public_data.df.sort_values(by="time_step", inplace=True)
            public_data.df.reset_index(inplace=True, drop=True)
            public_data.df.time_step = pd.to_datetime(public_data.df.time_step,
                                                      unit='s',
                                                      origin=pd.Timestamp('2018-01-01'))
            tsdata = public_data.get_tsdata(dt_col='time_step', target_col='cpu_usage')
            assert tsdata.df.shape == (61570, 4)
            assert set(tsdata.df.columns) == {'time_step', 'cpu_usage', 'mem_usage', 'id'}
            tsdata._check_basic_invariants()
