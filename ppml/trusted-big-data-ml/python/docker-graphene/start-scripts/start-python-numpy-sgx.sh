#!/bin/bash
SGX=1 ./pal_loader bash -c "python ./work/examples/test-numpy.py" | tee test-numpy-sgx.log
