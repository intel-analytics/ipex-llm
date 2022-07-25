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

echo "Running chronos tests TF1 and Deprecated API"
python -m pytest -v test/bigdl/chronos/autots/test_autotsestimator.py::TestAutoTrainer::test_fit_lstm_feature \
                    test/bigdl/chronos/autots/test_autotsestimator.py::TestAutoTrainer::test_fit_tcn_feature \
                    test/bigdl/chronos/autots/test_autotsestimator.py::TestAutoTrainer::test_fit_seq2seq_feature \
                    test/bigdl/chronos/autots/test_tspipeline.py::TestTSPipeline::test_seq2seq_tsppl_support_dataloader \
                    test/bigdl/chronos/autots/test_tspipeline.py::TestTSPipeline::test_tsppl_quantize_data_creator \
                    test/bigdl/chronos/autots/model/test_auto_lstm.py::TestAutoLSTM::test_onnx_methods \
                    test/bigdl/chronos/autots/model/test_auto_lstm.py::TestAutoLSTM::test_save_load \
                    test/bigdl/chronos/autots/model/test_auto_tcn.py::TestAutoTCN::test_onnx_methods \
                    test/bigdl/chronos/autots/model/test_auto_tcn.py::TestAutoTCN::test_save_load \
                    test/bigdl/chronos/autots/model/test_auto_seq2seq.py::TestAutoSeq2Seq::test_onnx_methods \
                    test/bigdl/chronos/autots/model/test_auto_seq2seq.py::TestAutoSeq2Seq::test_save_load \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_lstm_forecaster_onnx_methods \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_lstm_forecaster_quantization_onnx \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_lstm_forecaster_quantization_onnx_tuning \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_lstm_forecaster_distributed \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_lstm_forecaster_fit_loader \
                    test/bigdl/chronos/forecaster/test_lstm_forecaster.py::TestChronosModelLSTMForecaster::test_forecaster_from_tsdataset_data_loader_onnx \
                    test/bigdl/chronos/forecaster/test_seq2seq_forecaster.py::TestChronosModelSeq2SeqForecaster::test_s2s_forecaster_onnx_methods \
                    test/bigdl/chronos/forecaster/test_seq2seq_forecaster.py::TestChronosModelSeq2SeqForecaster::test_s2s_forecaster_distributed \
                    test/bigdl/chronos/forecaster/test_seq2seq_forecaster.py::TestChronosModelSeq2SeqForecaster::test_s2s_forecaster_fit_loader \
                    test/bigdl/chronos/forecaster/test_seq2seq_forecaster.py::TestChronosModelSeq2SeqForecaster::test_forecaster_from_tsdataset_data_loader_onnx \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_tcn_forecaster_onnx_methods \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_tcn_forecaster_quantization_onnx \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_tcn_forecaster_quantization_onnx_tuning \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_tcn_forecaster_distributed \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_tcn_forecaster_fit_loader \
                    test/bigdl/chronos/forecaster/test_tcn_forecaster.py::TestChronosModelTCNForecaster::test_forecaster_from_tsdataset_data_loader_onnx \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_nbeats_forecaster_onnx_methods \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_nbeats_forecaster_quantization_onnx \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_nbeats_forecaster_quantization_onnx_tuning \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_nbeats_forecaster_fit_loader \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_nbeats_forecaster_distributed \
                    test/bigdl/chronos/forecaster/test_nbeats_forecaster.py::TestChronosNBeatsForecaster::test_forecaster_from_tsdataset_data_loader_onnx

exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi

ray stop -f
