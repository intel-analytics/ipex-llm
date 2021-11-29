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

# The Clear BSD License

# Copyright (c) 2019 Carnegie Mellon University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted (subject to the limitations in the disclaimer below) provided that
# the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of Carnegie Mellon University nor the names of its contributors
#       may be used to endorse or promote products derived from this software without
#       specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import tempfile
import numpy as np
import pytest

from bigdl.chronos.simulator import DPGANSimulator
from bigdl.chronos.simulator.doppelganger.output import Output, OutputType, Normalization
from bigdl.orca.test_zoo_utils import ZooTestCase


def get_train_data():
    import os
    import io
    import shutil
    import urllib.request as req
    dfp = f'{os.getenv("FTP_URI")}/analytics-zoo-data/apps/doppelGANger_data/data_train_small.npz'
    fi = io.BytesIO()
    with req.urlopen(dfp) as dp:
        shutil.copyfileobj(dp, fi)
        fi.seek(0)
        df = np.load(fi)
    return df


class TestDoppelganer(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_output_value(self):
        attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=2),
                             Output(type_=OutputType.CONTINUOUS, dim=1,
                                    normalization=Normalization.MINUSONE_ONE)]
        assert set([val.type_.value for val in attribute_outputs]) == \
            set([val.type_.name for val in attribute_outputs])

        # illegal input.
        with pytest.raises(Exception):
            [Output(type_=OutputType.CONTINUOUS, dim=2, normalization=None)]

    def test_init_doppelganer(self):
        df = get_train_data()
        feature_outputs = [Output(type_=OutputType.CONTINUOUS,
                                  dim=1,
                                  normalization=Normalization.MINUSONE_ONE)]
        attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9),
                             Output(type_=OutputType.DISCRETE, dim=3),
                             Output(type_=OutputType.DISCRETE, dim=2)]

        doppelganger = DPGANSimulator(L_max=550,
                                      sample_len=10,
                                      feature_dim=1,
                                      num_real_attribute=3,
                                      num_threads=1)
        doppelganger.fit(data_feature=df['data_feature'],
                         data_attribute=df['data_attribute'],
                         data_gen_flag=df['data_gen_flag'],
                         feature_outputs=feature_outputs,
                         attribute_outputs=attribute_outputs,
                         epoch=2,
                         batch_size=32)

        feature, attribute, gen_flags, lengths = doppelganger.generate()
        assert feature.shape == (1, doppelganger.L_max, 1)
        assert attribute.shape == (1, df['data_attribute'].shape[-1])
        assert gen_flags.shape == (1, doppelganger.L_max) and (gen_flags[0, :] == 1).all()
        assert lengths[0] == doppelganger.L_max

        with tempfile.TemporaryDirectory() as tf:
            doppelganger.save(tf)
            doppelganger.load(tf)
        df.close()

        # illegal input
        df = get_train_data()
        feature_outputs = [Output(type_=OutputType.CONTINUOUS,
                           dim=1,
                           normalization=Normalization.MINUSONE_ONE)]
        attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9),
                             Output(type_=OutputType.DISCRETE, dim=3),
                             Output(type_=OutputType.DISCRETE, dim=2)]

        doppelganger = DPGANSimulator(L_max=551,
                                      sample_len=10,
                                      feature_dim=1,
                                      num_real_attribute=3,
                                      num_threads=1)

        with pytest.raises(RuntimeError):
            doppelganger.fit(data_feature=df['data_feature'],
                             data_attribute=df['data_attribute'],
                             data_gen_flag=df['data_gen_flag'],
                             feature_outputs=feature_outputs,
                             attribute_outputs=attribute_outputs)
        df.close()
