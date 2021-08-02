#
# Copyright 2018 Analytics Zoo Authors.
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
import re
import requests

import pandas as pd
from zoo.chronos.data.tsdataset import TSDataset

NETWORK_TRAFFIC_DATA = ['2018%02d' % i for i in range(1, 13)]\
    + ['2019%02d' % i for i in range(1, 13)]
BASE_URL = {'network_traffic': [f'http://mawi.wide.ad.jp/~agurim/dataset/{val}/{val}.agr'
                                for val in NETWORK_TRAFFIC_DATA],
            'AIOps': 'http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz'
            }


class PublicDataset:

    def __init__(self, name, path, redownload):
        self.name = name
        self.redownload = redownload

        self.url = BASE_URL[self.name]
        self.abspath = os.path.join(os.path.expanduser(path), self.name)
        self.data_path = os.path.join(self.abspath, self.name + '_data.csv')

    def get_public_data(self, chunk_size=1024):
        """
        Complete path stitching and download files.
        param chunk_size: Byte file_size of a single read, preferably an integer multiple of 2.
        """
        assert isinstance(chunk_size, int), "chunk_size must be int."
        if not os.path.exists(self.abspath):
            os.makedirs(self.abspath)

        if self.redownload:
            try:
                exists_file = os.listdir(self.abspath)
                _ = [os.remove(os.path.join(self.abspath, x))
                     for x in exists_file if x in NETWORK_TRAFFIC_DATA]
                os.remove(os.path.join(self.abspath, self.name + '_data.csv'))
            except Exception:
                raise OSError(
                    'File download is not completed, you should set redownload=False.')

        if not set(NETWORK_TRAFFIC_DATA).issubset(set(os.listdir(self.abspath))):
            if isinstance(BASE_URL[self.name], list):
                for val in self.url:
                    download(val, self.abspath, chunk_size)
            else:
                download(self.url, self.abspath, chunk_size)
        return self

    def preprocess_network_traffic(self):
        """
        preprocess_network_traffic will match the Starttime and endtime(avgrate, total)
        of data accordingto the regularity, and generate a csv file, the file name
        is network_traffic_data.csv.
        """
        _is_first_columns = True
        pattern = r"%Start.*?\((.*?)\)\n%%End.*?\((.*?)\)\n%Avg.*?\s(\d+\.\w+).*?\n%total:\s(\d+)"

        if not os.path.exists(self.data_path):
            for val in NETWORK_TRAFFIC_DATA:
                with open(os.path.join(self.abspath, val), 'r') as f:
                    content = f.read()
                    result = re.findall(pattern, content, re.DOTALL)
                    columns_name = ['StartTime', 'EndTime', 'AvgRate', 'total']
                raw_df_val = pd.DataFrame(columns=columns_name, data=result)
                raw_df_val.to_csv(self.data_path, mode='a',
                                  header=_is_first_columns, index=False, chunksize=256)
                _is_first_columns = False
        raw_df = pd.read_csv(self.data_path)
        self.df = pd.DataFrame(pd.to_datetime(raw_df.StartTime))
        raw_df.AvgRate.str[-4:].unique()
        self.df['AvgRate'] = raw_df.AvgRate.apply(lambda x: float(x[:-4]) if
                                                  x.endswith("Mbps") else float(x[:-4])*1000)
        self.df["total"] = raw_df["total"]
        return self

    def get_tsdata(self, dt_col, target_col, extra_feature=None, id_col=None):
        """
        param dt_col: same as tsdata.from_pandas.
        param target_col: same as tsdata.from_pandas.
        param extra_feature: same as tsdata.from_pandas.
        param id_col: same as tsdata.from_pandas.
        return tsdata_train,tsdata_valid,tsdata_test, The sample ratio is 0.8:0.1:0.1.
        """
        return TSDataset.from_pandas(self.df, dt_col=dt_col, target_col=target_col,
                                     extra_feature_col=extra_feature, id_col=id_col,
                                     with_split=True, val_ratio=0.1, test_ratio=0.1)


def download(url, path, chunk_size):
    """
    param url: File download source address, str or list.
    param path: File save path.default path/name/name_data.csv.
    """
    req = requests.get(url, stream=True)
    file_size, content_size = 0, int(req.headers['content-length'])
    assert req.status_code == 200, "download failure, please check the network."
    file_name = url.split('/')[-1].partition('.')[0]
    with open(os.path.join(path, file_name), 'wb') as f:
        for chunk in req.iter_content(1024 * chunk_size):
            if chunk:
                f.write(chunk)
                file_size += len(chunk)
                print('\r'+'file %s:%s%.2f%%' %
                      (file_name, '>' * int(file_size * 50 / content_size),
                       float(file_size / content_size * 100)), end='')
                f.flush()
        print('')
