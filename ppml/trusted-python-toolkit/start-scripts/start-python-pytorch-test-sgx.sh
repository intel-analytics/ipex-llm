export sgx_command="python /ppml/examples/pytorch_test.py"
gramine-sgx bash 2>&1 | tee pytorch_test.log