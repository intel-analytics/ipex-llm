#!/bin/bash
cd /ppml
./init.sh
export sgx_command="bash examples/numpy/hello-numpy.py"
gramine-sgx bash 2>&1

