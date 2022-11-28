set -e

bash $ANALYTICS_ZOO_ROOT/python/chronos/dev/release/release.sh linux default false
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch --force-reinstall
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/nano/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/nano/dist/${whl_name}[pytorch,inference]
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/${whl_name}[pytorch]

cd $ANALYTICS_ZOO_ROOT/python/chronos/benchmark/forecaster_inference/

echo "Chronos_Perf: Running TCNForecaster Inference Pytorch Baseline"
source bigdl-nano-init
python forecaster_inference.py --name "TCNForecaster Inference Pytorch Baseline" --accelerator "pytorch"
source bigdl-nano-unset-env

echo "Chronos_Perf: Running TCNForecaster Inference with ONNX"
source bigdl-nano-init
python forecaster_inference.py --name "TCNForecaster Inference with ONNX" --accelerator "onnx"
source bigdl-nano-unset-env

echo "Chronos_Perf: Running TCNForecaster Inference with OpenVINO"
source bigdl-nano-init
python forecaster_inference.py --name "TCNForecaster Inference with OpenVINO" --accelerator "openvino"
source bigdl-nano-unset-env
