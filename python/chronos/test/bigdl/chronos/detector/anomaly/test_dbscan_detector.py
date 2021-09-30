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

import pytest
import numpy as np
from bigdl.orca.test_zoo_utils import ZooTestCase

from bigdl.chronos.detector.anomaly.dbscan_detector import DBScanDetector


class TestDBScanDetector(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def create_data(self):
        cycles = 5
        time = np.arange(0, cycles * np.pi, 0.2)
        data = np.sin(time)
        data[3] += 10
        data[5] -= 2
        data[10] += 5
        data[17] -= 3
        return data

    def test_dbscan_fit_score(self):
        y = self.create_data()
        ad = DBScanDetector(eps=0.1, min_samples=6)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        # directly use dbscan may cause high local false positive/negatives
        # so the detected anomalies is probably more than the actual ones
        assert len(anomaly_indexes) >= 4

    def test_corner_cases(self):
        ad = DBScanDetector(eps=0.1, min_samples=6)
        with pytest.raises(RuntimeError):
            ad.score()
        with pytest.raises(RuntimeError):
            ad.anomaly_indexes()
        y = self.create_data()
        y = y[:-1].reshape(2, -1)
        with pytest.raises(ValueError):
            ad.fit(y)
