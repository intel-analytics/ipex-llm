#!/bin/bash
cd /ppml
export sgx_command="python ./examples/APP.py"
gramine-sgx bash 2>&1 | tee test-flask-sgx.log
