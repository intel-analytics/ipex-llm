#!/bin/bash
cd /ppml/trusted-big-data-ml
export spark_commnd="python ./work/examples/test-numpy.py"
gramine-sgx bash 2>&1 | tee test-numpy-sgx.log
cat test-numpy-sgx.log | egrep -a "dot"
