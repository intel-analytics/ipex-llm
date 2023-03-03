cd /ppml
./init.sh
export sgx_command="/opt/tritonserver/bin/tritonserver --model-repository=/ppml/work/data/model/tritonserver/ --model-load-thread-count 5"
gramine-sgx bash 2>&1 | tee tritonserver-sgx.log

