#!/usr/bin/env bash

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


cd "`dirname $0`"
cd ../..

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python
if [ -z "${OMP_NUM_THREADS}" ]; then
    export OMP_NUM_THREADS=1
fi

ray stop -f

RUN_PART1=0
RUN_PART2=0
RUN_PART3=0
RUN_PART4=0
if [ $1 = 1 ]; then
RUN_PART1=1
elif [ $1 = 2 ]; then
RUN_PART2=1
elif [ $1 = 3 ]; then
RUN_PART3=1
else
RUN_PART4=1
fi

if [ $RUN_PART1 = 1 ]; then
echo "Running chronos tests Part 1"
python -m pytest -v test/bigdl/chronos/forecaster/test_lstm_forecaster.py \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py \
                    test/bigdl/chronos/forecaster/test_seq2seq_forecaster.py \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

if [ $RUN_PART2 = 1 ]; then
echo "Running chronos tests Part 2"
python -m pytest -v test/bigdl/chronos/forecaster \
       -k "not TestChronosModelLSTMForecaster and \
           not TestChronosNBeatsForecaster and \
           not TestChronosModelSeq2SeqForecaster and \
           not TestChronosModelTCNForecaster and \
           not test_forecast_tcmf_distributed and \
           not test_tcn_keras_forecaster_quantization"
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
python -m pytest -s test/bigdl/chronos/forecaster/tf/\
test_tcn_keras_forecaster.py::TestTCNForecaster::test_tcn_keras_forecaster_quantization
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

if [ $RUN_PART3 = 1 ]; then
echo "Running chronos tests Part 3"
python -m pytest -v test/bigdl/chronos/detector\
                    test/bigdl/chronos/metric \
                    test/bigdl/chronos/model \
                    test/bigdl/chronos/pytorch \
                    test/bigdl/chronos/simulator \
       -k "not test_ae_fit_score_unrolled"
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

if [ $RUN_PART4 = 1 ]; then
echo "Running chronos tests Part 4"
python -m pytest -v test/bigdl/chronos/autots\
                    test/bigdl/chronos/data\
                    test/bigdl/chronos/aiops
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

ray stop -f

