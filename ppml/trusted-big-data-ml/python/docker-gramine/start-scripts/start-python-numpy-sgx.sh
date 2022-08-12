#!/bin/bash
cd /ppml/trusted-big-data-ml
./clean.sh
gramine-argv-serializer bash -c "python ./work/examples/test-numpy.py" > secured_argvs
./init.sh
gramine-sgx bash 2>&1 | tee test-numpy-sgx.log
cat test-numpy-sgx.log | egrep -a "dot"
