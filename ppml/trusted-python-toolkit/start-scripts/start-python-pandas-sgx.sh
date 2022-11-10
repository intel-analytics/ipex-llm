#!/bin/bash

cd /ppml
bash /ppml/examples/pandas/CLI.sh -n 1m -p native
export sgx_command="bash /ppml/examples/pandas/CLI.sh -n 1m -p sgx"
gramine-sgx bash 2>&1 | tee test-pandas-sgx.log

bash /ppml/examples/pandas/CLI.sh -n 10m -p native
export sgx_command="bash /ppml/examples/pandas/CLI.sh -n 10m -p sgx"
gramine-sgx bash 2>&1 | tee test-pandas-sgx.log

bash /ppml/examples/pandas/CLI.sh -n full -p native
export sgx_command="bash /ppml/examples/pandas/CLI.sh -n full -p sgx"
gramine-sgx bash 2>&1 | tee test-pandas-sgx.log

