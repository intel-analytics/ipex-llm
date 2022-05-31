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
from bigdl.chronos.data.utils.public_dataset import PublicDataset


def get_public_dataset(name, path='~/.chronos/dataset', redownload=False, **kwargs):
    """
    Get public dataset.

    >>> from bigdl.chronos.data.repo_dataset import get_public_dataset
    >>> tsdata_network_traffic = get_public_dataset(name="network_traffic")

    :param name: str, public dataset name, e.g. "network_traffic".
           We only support network_traffic, AIOps, fsi, nyc_taxi, uci_electricity,
           uci_electricity_wide.
    :param path: str, download path, the value defatults to "~/.chronos/dataset/".
    :param redownload: bool, if redownload the raw dataset file(s).
    :param kwargs: extra arguments passed to initialize the tsdataset,
           including with_split, val_ratio and test_ratio.
    """
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(isinstance(name, str) and isinstance(path, str),
                      "Name and path must be string.")

    if name.lower().strip() == 'network_traffic':
        return PublicDataset(name='network_traffic',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_network_traffic()\
                                      .get_tsdata(dt_col='StartTime',
                                                  target_col=['AvgRate', 'total'])
    elif name.lower().strip() == 'aiops':
        return PublicDataset(name='AIOps',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_AIOps()\
                                      .get_tsdata(dt_col='time_step',
                                                  target_col=['cpu_usage'])
    elif name.lower().strip() == 'fsi':
        return PublicDataset(name='fsi',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_fsi()\
                                      .get_tsdata(dt_col='ds',
                                                  target_col=['y'])
    elif name.lower().strip() == 'nyc_taxi':
        return PublicDataset(name='nyc_taxi',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_nyc_taxi()\
                                      .get_tsdata(dt_col='timestamp',
                                                  target_col=['value'])
    elif name.lower().strip() == 'uci_electricity':
        return PublicDataset(name='uci_electricity',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_uci_electricity()\
                                      .get_tsdata(dt_col='timestamp',
                                                  target_col=['value'],
                                                  id_col='id')
    elif name.lower().strip() == 'uci_electricity_wide':
        target = []
        for i in range(370):
            target.append('MT_'+str(i+1).zfill(3))
        return PublicDataset(name='uci_electricity_wide',
                             path=path,
                             redownload=redownload,
                             **kwargs).get_public_data()\
                                      .preprocess_uci_electricity_wide()\
                                      .get_tsdata(dt_col='timestamp',
                                                  target_col=target)
    else:
        invalidInputError(False,
                          "Only network_traffic, AIOps, fsi, nyc_taxi, uci_electricity"
                          " uci_electricity_wide"
                          f"are supported in Chronos built-in dataset, while get {name}.")
