#!/bin/bash
cd /ppml
export sgx_command="python ./examples/pandas/test-pandas.py"
gramine-sgx bash 2>&1 | tee test-numpy-sgx.log

