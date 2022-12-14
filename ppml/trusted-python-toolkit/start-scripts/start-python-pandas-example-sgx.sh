#!/bin/bash

cd /ppml
./init.sh
export sgx_command="bash /ppml/examples/pandas/hello-pandas.py -d $dataset -p sgx"
gramine-sgx bash 2>&1

