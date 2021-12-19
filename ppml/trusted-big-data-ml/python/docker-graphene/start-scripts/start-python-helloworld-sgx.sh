#!/bin/bash
cd /ppml/trusted-big-data-ml
SGX=1 ./pal_loader bash -c "python ./work/examples/helloworld.py" | tee test-helloworld-sgx.log && \
	cat test-helloworld-sgx.log | egrep -a "Hello"
