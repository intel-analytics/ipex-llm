export sgx_command="python /ppml/examples/sklearn_test.py"
gramine-sgx bash 2>&1 | tee sklearn_test.log