#!/bin/bash
cd /ppml/trusted-big-data-ml
./clean.sh
gramine-argv-serializer bash -c "python ./work/examples/helloworld.py" > secured_argvs
./init.sh
gramine-sgx bash 2>&1 | tee test-helloworld-sgx.log
cat test-helloworld-sgx.log | egrep -a "Hello"
