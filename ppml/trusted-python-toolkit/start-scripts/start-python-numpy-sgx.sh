#!/bin/bash
cd /ppml
export sgx_command="python ./examples/numpy/test-numpy.py"
gramine-sgx bash 2>&1 | tee test-numpy-sgx.log

