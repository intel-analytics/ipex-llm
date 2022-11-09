set -e

bash $ANALYTICS_ZOO_ROOT/python/chronos/dev/release/release.sh linux default false
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch
whl_name = `ls $ANALYTICS_ZOO_ROOT/python/nano/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/nano/dist/${whl_name}[pytorch,inference]
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/${whl_name}[pytorch]

cd $ANALYTICS_ZOO_ROOT/python/chronos/benchmark/forecaster_inference/

echo "Chronos_Perf: Running TCNForecaster Inference Pytorch Baseline"
python forecaster_training.py --name "TCNForecaster Inference Pytorch Baseline" --accelerator "pytorch"

echo "Chronos_Perf: Running TCNForecaster Inference with ONNX"
python forecaster_training.py --name "TCNForecaster Inference with ONNX" --accelerator "onnx"

echo "Chronos_Perf: Running TCNForecaster Inference with OpenVINO"
python forecaster_training.py --name "TCNForecaster Inference with OpenVINO" --accelerator "openvino"
