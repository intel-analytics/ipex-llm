export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_BENCHMARK_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/benchmark
export WORKLOAD_DIR=${NANO_BENCHMARK_DIR}/torch_inference_optimizer

set -e

# install dependencies
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch,inference

# set nano's environment variables
source bigdl-nano-unset-env
source bigdl-nano-init

# cd workload directory
cd $WORKLOAD_DIR

python optimize.py

options="original fp32_channels_last fp32_ipex fp32_ipex_channels_last bf16 \
         bf16_channels_last bf16_ipex bf16_ipex_channels_last int8 int8_ipex \
         jit_fp32 jit_bf16 jit_fp32_ipex jit_fp32_ipex_channels_last \
         jit_bf16_ipex jit_bf16_ipex_channels_last openvino_fp32 openvino_int8 \
         onnxruntime_fp32 onnxruntime_int8_qlinear onnxruntime_int8_integer"

for option in $options;
do
    bash benchmark.sh 1 4 $option
done

for option in $options;
do
    bash benchmark.sh 6 4 $option
done
