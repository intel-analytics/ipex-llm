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
# This file is adapted from the DeepGlo Project. https://github.com/rajatsen91/deepglo
#
# Note: This license has also been called the "New BSD License" or "Modified BSD License". See also
# the 2-clause BSD License.
#
# Copyright (c) 2019 The DeepGLO Project.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import pandas as pd
import numpy as np
from packaging import version


class TimeCovariates(object):
    def __init__(self, start_date, num_ts=100, freq="H", normalized=True):
        self.start_date = start_date
        self.num_ts = num_ts
        self.freq = freq
        self.normalized = normalized
        self.dti = pd.date_range(self.start_date, periods=self.num_ts, freq=self.freq)

    def _minute_of_hour(self):
        minutes = np.array(self.dti.minute, dtype=np.float32)
        if self.normalized:
            minutes = minutes / 59.0 - 0.5
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dti.hour, dtype=np.float32)
        if self.normalized:
            hours = hours / 23.0 - 0.5
        return hours

    def _day_of_week(self):
        dayWeek = np.array(self.dti.dayofweek, dtype=np.float32)
        if self.normalized:
            dayWeek = dayWeek / 6.0 - 0.5
        return dayWeek

    def _day_of_month(self):
        dayMonth = np.array(self.dti.day, dtype=np.float32)
        if self.normalized:
            dayMonth = dayMonth / 30.0 - 0.5
        return dayMonth

    def _day_of_year(self):
        dayYear = np.array(self.dti.dayofyear, dtype=np.float32)
        if self.normalized:
            dayYear = dayYear / 364.0 - 0.5
        return dayYear

    def _month_of_year(self):
        monthYear = np.array(self.dti.month, dtype=np.float32)
        if self.normalized:
            monthYear = monthYear / 11.0 - 0.5
        return monthYear

    def _week_of_year(self):
        weekYear = np.array(pd.Int64Index(self.dti.isocalendar().week), dtype=np.float32) if\
            version.parse(pd.__version__) >= version.parse("1.1.0") else\
            np.array(self.dti.weekofyear, dtype=np.float32)
        if self.normalized:
            weekYear = weekYear / 51.0 - 0.5
        return weekYear

    def get_covariates(self):
        MOH = self._minute_of_hour().reshape(1, -1)
        HOD = self._hour_of_day().reshape(1, -1)
        DOM = self._day_of_month().reshape(1, -1)
        DOW = self._day_of_week().reshape(1, -1)
        DOY = self._day_of_year().reshape(1, -1)
        MOY = self._month_of_year().reshape(1, -1)
        WOY = self._week_of_year().reshape(1, -1)

        all_covs = [MOH, HOD, DOM, DOW, DOY, MOY, WOY]

        return np.vstack(all_covs)
