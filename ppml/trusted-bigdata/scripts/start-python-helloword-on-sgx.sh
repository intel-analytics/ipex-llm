#!/bin/bash
cd /ppml
export sgx_command="python ./examples/helloworld.py"
gramine-sgx bash 2>&1 | tee test-helloworld-sgx.log
cat test-helloworld-sgx.log | egrep -a "Hello"
