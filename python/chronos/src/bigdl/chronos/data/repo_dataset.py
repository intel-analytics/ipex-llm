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
import numpy as np
import pandas as pd
import time
from bigdl.chronos.data.utils.public_dataset import PublicDataset
from bigdl.chronos.data.tsdataset import TSDataset


def get_public_dataset(name, path='~/.chronos/dataset', redownload=False, **kwargs):
    """
    Get public dataset.

    >>> from bigdl.chronos.data import get_public_dataset
    >>> tsdata_network_traffic = get_public_dataset(name="network_traffic")

    :param name: str, public dataset name, e.g. "network_traffic".
           We only support network_traffic, AIOps, fsi, nyc_taxi, uci_electricity,
           uci_electricity_wide, tsinghua_electricity
    :param path: str, download path, the value defatults to "~/.chronos/dataset/".
    :param redownload: bool, if redownload the raw dataset file(s).
    :param kwargs: extra arguments passed to initialize the tsdataset,
           including with_split, val_ratio and test_ratio.
    """
    from bigdl.nano.utils.common import invalidInputError
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
    elif name.lower().strip() == 'tsinghua_electricity':
        target = []
        for i in range(0, 320):
            target.append(str(i))
        target.append("OT")
        dataset = PublicDataset(name='tsinghua_electricity',
                                path=path,
                                redownload=False,
                                **kwargs).preprocess_tsinghua_electricity()\
                                         .get_tsdata(dt_col='date',
                                                     target_col=target)
        return dataset

    else:
        invalidInputError(False,
                          "Only network_traffic, AIOps, fsi, nyc_taxi, uci_electricity"
                          " uci_electricity_wide "
                          f"are supported in Chronos built-in dataset, while get {name}.")


def gen_synthetic_data(len=10000, sine_amplitude=10.0, angular_freq=0.01,
                       noise_amplitude=0.01, noise_scale=1.0, seed=1,
                       time_freq="D", **kwargs):
    """
    Generate dataset according to sine function with a Gaussian noise.
    Datetime is generated according to `time_freq` with the current time as endtime.

    >>> from bigdl.chronos.data import gen_synthetic_data
    >>> tsdata_gen = gen_synthetic_data()

    :param len: int, the number indicates the dataset size. Default to 10000.
    :param sine_amplitude: float, the number indicates amplitude of the sine function.
           Default to 10.0.
    :param angular_freq: float, the number indicates angular frequency of the sine function.
           Default to 0.01.
    :param noise_amplitude: float, the number indicates amplitude of the Gaussian noise.
           Default to 0.01.
    :param noise_scale: float, the number indicates the standard deviation of the Gaussian noise
           while the mean is set to 0. Default to 1.0.
    :param seed: int, random seed for generating Gaussian noise. Default to 1.
    :param time_freq: str, the frequency of the generated dataframe, default to 'D'(calendar day
           frequency). The frequency can be anything from the pandas list of frequency strings here:
           https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :param kwargs: extra arguments passed to initialize the tsdataset,
           including with_split, val_ratio and test_ratio.

    :return: a TSDataset instance when with_split is set to False,
             three TSDataset instances when with_split is set to True.
    """
    from bigdl.chronos.data.utils.utils import _check_type
    _check_type(len, "len", int)
    _check_type(sine_amplitude, "sine_amplitude", float)
    _check_type(angular_freq, "angular_freq", float)
    _check_type(noise_amplitude, "noise_amplitude", float)
    _check_type(noise_scale, "noise_scale", float)
    _check_type(seed, "seed", int)
    _check_type(time_freq, "time_freq", str)

    gen_x = np.linspace(0, len * angular_freq, len)
    np.random.seed(seed)
    gen_y = (sine_amplitude * np.sin(gen_x)
             + noise_amplitude * np.random.normal(0, noise_scale, len))
    df = pd.DataFrame(gen_y, columns=["target"])
    endtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    df.insert(0, "datetime", pd.date_range(end=endtime, periods=len, freq=time_freq))

    return TSDataset.from_pandas(df, dt_col="datetime", target_col="target", **kwargs)
