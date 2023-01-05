cd /ppml
./init.sh
export sgx_command="/opt/tritonserver/bin/tritonserver --model-repository /ppml/work/data/try/python_backend/models/"
gramine-sgx bash 2>&1 | tee backend-sgx.log

