#!/bin/bash
cd /ppml
./init.sh
export sgx_command="python ./examples/flask/APP.py"
gramine-sgx bash 2>&1 | tee test-flask-sgx.log
