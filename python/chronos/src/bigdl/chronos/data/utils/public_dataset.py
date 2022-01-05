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
import requests

import tqdm
import pandas as pd
import numpy as np
from bigdl.chronos.data.tsdataset import TSDataset


DATASET_NAME = {'network_traffic': ['2018%02d.agr' % i for i in range(1, 13)]
                + ['2019%02d.agr' % i for i in range(1, 13)],
                'AIOps': ['machine_usage.tar.gz'],
                'fsi': ['individual_stocks_5yr.zip'],
                'nyc_taxi': ['nyc_taxi.csv'],
                'uci_electricity': ['LD2011_2014.txt.zip']
                }
BASE_URL = \
    {'network_traffic':
     [f'http://mawi.wide.ad.jp/~agurim/dataset/{val[:-4]}/{val}'
      for val in DATASET_NAME['network_traffic']],
     'AIOps':
     ['http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz'],
     'fsi':
     ['https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip'],
     'nyc_taxi':
     ['https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv'],
     'uci_electricity':
     ['https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip']}


class PublicDataset:

    def __init__(self, name, path, redownload, **kwargs):
        self.name = name
        self.redownload = redownload
        self.with_split = kwargs.get('with_split', True)
        self.val_ratio = kwargs.get('val_ratio', 0.1)
        self.test_ratio = kwargs.get('test_ratio', 0.1)

        self.url = BASE_URL[self.name]
        self.dir_path = os.path.join(os.path.expanduser(path), self.name)
        self.final_file_path = os.path.join(self.dir_path, self.name + '_data.csv')

    def get_public_data(self, chunk_size=1024):
        '''
        Complete path stitching and download files.
        param chunk_size: Byte size of a single read, preferably an integer multiple of 2.
        '''
        assert isinstance(chunk_size, int), "chunk_size must be int."
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # check local file exists.
        if self.redownload:
            for val in self.url:
                download(val, self.dir_path, chunk_size)
        else:
            if not os.path.exists(self.final_file_path) and \
                    not set(DATASET_NAME[self.name]).issubset(set(os.listdir(self.dir_path))):
                for val in self.url:
                    download(val, self.dir_path, chunk_size)
        return self

    def preprocess_network_traffic(self):
        '''
        return data that meets the minimum requirements of tsdata.
        '''
        _is_first_columns = True
        pattern = r"%Start.*?\((.*?)\)\n%%End.*?\((.*?)\)\n%Avg.*?\s(\d+\.\w+).*?\n%total:\s(\d+)"
        columns_name = ['StartTime', 'EndTime', 'AvgRate', 'total']

        if not os.path.exists(self.final_file_path):
            import re
            for val in DATASET_NAME[self.name]:
                with open(os.path.join(self.dir_path, val), 'r') as f:
                    content = f.read()
                    result = re.findall(pattern, content, re.DOTALL)
                net_temp_df = pd.DataFrame(columns=columns_name,
                                           data=result)
                net_temp_df.to_csv(self.final_file_path,
                                   mode='a',
                                   header=_is_first_columns,
                                   index=False,
                                   chunksize=1024)
                _is_first_columns = False

        net_raw_df = pd.read_csv(self.final_file_path)
        net_raw_df.AvgRate.str[-4:].unique()
        self.df = pd.DataFrame(pd.to_datetime(net_raw_df.StartTime))
        self.df['AvgRate'] = net_raw_df.AvgRate.apply(lambda x: float(x[:-4]) if
                                                      x.endswith("Mbps") else float(x[:-4])*1000)
        self.df["total"] = net_raw_df["total"]
        return self

    def preprocess_AIOps(self):
        '''
        return data that meets the minimum requirements of tsdata.
        '''
        _is_first_columns = True
        file_path = os.path.join(os.path.expanduser(self.dir_path), DATASET_NAME[self.name][0])
        download_csv_name = DATASET_NAME[self.name][0].split('.')[0] + '.csv'

        try:
            if not os.path.exists(os.path.join(self.dir_path, download_csv_name)):
                import tarfile
                tar = tarfile.open(file_path, 'r:gz')
                tar.extractall(os.path.expanduser(self.dir_path))
        except Exception:
            raise FileExistsError('file extractall failure, set redownload=True.')

        aio_raw_df = pd.read_csv(os.path.join(self.dir_path,
                                              download_csv_name),
                                 header=None,
                                 usecols=[0, 1, 2, 3],
                                 names=["id", "time_step", "cpu_usage", "mem_usage"],
                                 na_filter=False,
                                 chunksize=4096,
                                 low_memory=False)

        if not os.path.exists(self.final_file_path):
            for val in aio_raw_df:
                val.loc[val.id.eq('m_1932')].\
                    to_csv(self.final_file_path,
                           mode='a',
                           header=["id", "time_step", "cpu_usage", "mem_usage"]
                           if _is_first_columns else None,
                           index=False)
                _is_first_columns = False

        self.df = pd.read_csv(self.final_file_path, usecols=[1, 2, 3])
        self.df.sort_values(by="time_step", inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df["time_step"] = \
            pd.to_datetime(self.df["time_step"], unit='s', origin=pd.Timestamp('2018-01-01'))
        return self

    def preprocess_fsi(self):
        """
        return data that meets the minimum requirements of tsdata.
        """
        _is_first_columns = True
        if not os.path.exists(self.final_file_path):
            import zipfile
            zip_file = zipfile.ZipFile(os.path.join(
                                       os.path.expanduser(self.dir_path),
                                       DATASET_NAME[self.name][0]))
            zip_file.extractall(os.path.join(os.path.expanduser(self.dir_path)))
            download_file = os.path.join(self.dir_path, DATASET_NAME[self.name][0].split('.')[0])
            local_file_list = os.listdir(download_file)

            for val in local_file_list:
                fsi_raw_df = pd.read_csv(os.path.join(download_file, val),
                                         names=['date', 'open', 'high',
                                                'low', 'close', 'volume', 'Name'])
                fsi_raw_df = fsi_raw_df.loc[fsi_raw_df.Name.eq('MMM')]
                fsi_raw_df.to_csv(self.final_file_path,
                                  header=['ds', 'open', 'high',
                                          'low', 'close', 'y', 'Name']
                                  if _is_first_columns else None,
                                  index=False,
                                  mode='a')
                _is_first_columns = False

        self.df = pd.read_csv(self.final_file_path, usecols=[0, 5], parse_dates=[0])
        self.df.ds = pd.to_datetime(self.df.ds)

        return self

    def preprocess_nyc_taxi(self):
        '''
        Return data that meets the minimum requirements of tsdata.
        '''
        raw_csv_name = os.path.join(self.dir_path, DATASET_NAME[self.name][0])
        if not os.path.exists(self.final_file_path):
            with open(raw_csv_name, 'rb') as src, open(self.final_file_path, 'wb') as dst:
                dst.write(src.read())
        self.df = pd.read_csv(self.final_file_path, parse_dates=['timestamp'])
        return self

    def preprocess_uci_electricity(self):
        '''
        return data that meets the minimum requirements of tsdata.
        '''
        if not os.path.exists(self.final_file_path):
            import zipfile
            zip_file = zipfile.ZipFile(os.path.join(
                                       os.path.expanduser(self.dir_path),
                                       DATASET_NAME[self.name][0]))
            zip_file.extractall(os.path.join(os.path.expanduser(self.dir_path)))
            download_file = os.path.join(self.dir_path, DATASET_NAME[self.name][0].split('.')[0])
            os.rename(download_file+'.txt', self.final_file_path)
        df = pd.read_csv(self.final_file_path,
                         delimiter=';',
                         parse_dates=['Unnamed: 0'],
                         low_memory=False)
        self.df = pd.melt(df,
                          id_vars=['Unnamed: 0'],
                          value_vars=df.T.index[1:])\
                    .rename(columns={'Unnamed: 0': 'timestamp', 'variable': 'id'})
        self.df.value = self.df.value.apply(lambda x: str(x).replace(",", "")).astype(np.float32)
        return self

    def get_tsdata(self,
                   dt_col,
                   target_col,
                   extra_feature=None,
                   id_col=None):
        """
        param dt_col: same as tsdata.from_pandas.
        param target_col: same as tsdata.from_pandas.
        param extra_feature: same as tsdata.from_pandas.
        param id_col: same as tsdata.from_pandas.
        return tsdata.
        """
        if self.with_split:
            return TSDataset.from_pandas(self.df,
                                         dt_col=dt_col,
                                         target_col=target_col,
                                         extra_feature_col=extra_feature,
                                         id_col=id_col,
                                         with_split=self.with_split,
                                         val_ratio=self.val_ratio,
                                         test_ratio=self.test_ratio)
        else:
            return TSDataset.from_pandas(self.df,
                                         dt_col=dt_col,
                                         target_col=target_col,
                                         extra_feature_col=extra_feature,
                                         id_col=id_col)


def download(url, path, chunk_size):
    """
    param url: File download source address, str or list.
    param path: File save path, default path/name/name_data.csv.
    """
    req = requests.get(url, stream=True)
    file_size = int(req.headers['content-length'])
    assert req.status_code == 200, "download failure, please check the network."
    file_name = url.split('/')[-1]
    pbar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name)
    with open(os.path.join(path, file_name), 'wb') as f:
        for chunk in req.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(chunk_size)
                f.flush()
    pbar.close()
