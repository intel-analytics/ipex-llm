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

# The MIT License (MIT)

# Copyright (c) 2015 David Coles

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import datetime
import json
from typing import Optional, Union
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests

from bigdl.nano.utils.common import invalidInputError

Timestamp = Union[str, float, datetime.datetime]  # RFC-3339 string or a Unix timestamp in seconds
Duration = Union[str, datetime.timedelta]  # Prometheus duration string
Matrix = pd.DataFrame
Vector = pd.Series
Scalar = np.float64
String = str


class Prometheus:
    def __init__(self, api_url: str, http: Optional[requests.Session] = None):
        """
        Create Prometheus client.

        :param api_url: URL of Prometheus server.
        :param http: Requests Session to use for requests. Optional.
        """
        self.http = http or requests.Session()
        self.api_url = api_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.http.close()

    def query_range(self, query: str, start: Timestamp, end: Timestamp,
                    step: Union[Duration, float],
                    timeout: Optional[Duration] = None) -> Matrix:
        """
        Evaluates an expression query over a range of time.

        :param query: Prometheus expression query string.
        :param start: Start timestamp.
        :param end: End timestamp.
        :param step: Query resolution step width in `duration` format or float number of seconds.
        :param timeout: Evaluation timeout. Optional.
        :return: Pandas DataFrame.
        """
        params = {'query': query, 'start': _timestamp(start), 'end': _timestamp(end),
                  'step': _duration(step)}

        if timeout is not None:
            params['timeout'] = _duration(timeout)

        return to_pandas(self._do_query('api/v1/query_range', params))

    def _do_query(self, path: str, params: dict) -> dict:
        resp = self.http.get(urljoin(self.api_url, path), params=params)
        if resp.status_code not in [400, 422, 503]:
            resp.raise_for_status()

        response = resp.json()
        if response['status'] != 'success':
            invalidInputError(False, "Fail to collect data from Prometheus! "
                              "{errorType}: {error}".format_map(response))

        return response['data']


def to_pandas(data: dict) -> Union[Matrix, Vector, Scalar, String]:
    """Convert Prometheus data object to Pandas object."""
    result_type = data['resultType']
    if result_type == 'vector':
        return pd.Series((np.float64(r['value'][1]) for r in data['result']),
                         index=(metric_name(r['metric']) for r in data['result']))
    elif result_type == 'matrix':
        return pd.DataFrame({
            metric_name(r['metric']):
                pd.Series((np.float64(v[1]) for v in r['values']),
                          index=(pd.Timestamp(v[0], unit='s') for v in r['values']))
            for r in data['result']})
    elif result_type == 'scalar':
        return np.float64(data['result'])
    elif result_type == 'string':
        return data['result']
    else:
        invalidInputError(False, f"The collected Prometheus data is unknown type {result_type}.")


def metric_name(metric: dict) -> str:
    """Convert metric labels to standard form."""
    name = metric.get('__name__', '')
    labels = ','.join(('{}={}'.format(k, json.dumps(v)) for k, v in metric.items()
                      if k != '__name__'))
    return '{0}{{{1}}}'.format(name, labels)


def _timestamp(value):
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    else:
        return value


def _duration(value):
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    else:
        return value


def GetRangeDataframe(prometheus_url, query_list, starttime, endtime, step, columns, **kwargs):
    '''
    Convert the Prometheus data over the specified time period to dataframe and confirm
    dt_col, target_col, id_col and extra_feature_col.

    Return dataframe and col names.
    '''
    from bigdl.chronos.data.utils.utils import _check_type
    _check_type(prometheus_url, "prometheus_url", str)
    _check_type(starttime, "starttime", (str, float))
    _check_type(endtime, "endtime", (str, float))
    _check_type(step, "step", (str, float))

    # Generate dataframe according query_list
    pro_client = Prometheus(prometheus_url)
    pro_df = pd.DataFrame()
    for query in query_list:
        query_df = pro_client.query_range(query, starttime, endtime, step, **kwargs)
        # TODO: whatif pro_df and query_df has different length
        # TODO: repair missing value when query Prometheus client
        pro_df = pd.concat([pro_df, query_df], axis=1)

    df = pd.DataFrame(columns=pro_df.columns.tolist())
    df.insert(0, "datetime", pro_df.index)
    for col in pro_df.columns.tolist():
        df[col] = pro_df[col].values

    # Clean columns
    output_columns = {"dt_col": "datetime",
                      "target_col": pro_df.columns.tolist(),
                      "id_col": None,
                      "extra_feature_col": None}
    output_col_list = ["datetime"]
    for col in ["target_col", "id_col", "extra_feature_col"]:
        # Check whether columns specified by user exist
        invalidInputError(len(columns[col]) == 0 or
                          set(columns[col]).issubset(df.columns.tolist()),
                          "The input " + col + " is not found in collected Prometheus data.")

        # If users specify target_col/id_col/extra_feature_col, update values in output_columns[]
        if len(columns[col]) == 1:
            output_columns[col] = columns[col][0]  # id_col must be str
        elif len(columns[col]) > 1:
            output_columns[col] = columns[col]

        # According to output_columns, confirm columns to be remained in df
        if output_columns[col] is not None:
            if isinstance(output_columns[col], list):
                output_col_list = output_col_list + output_columns[col]
            else:
                output_col_list.append(output_columns[col])

    df = df.loc[:, output_col_list]
    return df, output_columns
