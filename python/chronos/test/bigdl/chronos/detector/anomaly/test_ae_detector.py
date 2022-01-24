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
from unittest import TestCase

from bigdl.chronos.detector.anomaly.ae_detector import AEDetector


class TestAEDetector(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def create_data(self):
        cycles = 10
        time = np.arange(0, cycles * np.pi, 0.01)
        data = np.sin(time)
        data[600:800] = 10
        return data

    def test_ae_fit_score_rolled_keras(self):
        y = self.create_data()
        ad = AEDetector(roll_len=314)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    def test_ae_fit_score_rolled_pytorch(self):
        y = self.create_data()
        ad = AEDetector(roll_len=314, backend="torch")
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    def test_ae_fit_score_unrolled(self):
        y = self.create_data()
        ad = AEDetector(roll_len=0)
        ad.fit(y)
        anomaly_scores = ad.score()
        assert len(anomaly_scores) == len(y)
        anomaly_indexes = ad.anomaly_indexes()
        assert len(anomaly_indexes) == int(ad.ratio * len(y))

    def test_corner_cases(self):
        y = self.create_data()
        ad = AEDetector(roll_len=314, backend="dummy")
        with pytest.raises(ValueError):
            ad.fit(y)
        ad = AEDetector(roll_len=314)
        with pytest.raises(RuntimeError):
            ad.score()
        y = np.array([1])
        with pytest.raises(ValueError):
            ad.fit(y)
        y = self.create_data()
        y = y.reshape(2, -1)
        with pytest.raises(ValueError):
            ad.fit(y)
