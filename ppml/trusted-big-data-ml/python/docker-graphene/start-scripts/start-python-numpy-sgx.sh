#!/bin/bash
cd /ppml/trusted-big-data-ml
SGX=1 ./pal_loader bash -c "python ./work/examples/test-numpy.py" | tee test-numpy-sgx.log && \
	cat test-numpy-sgx.log | egrep -a "dot"
