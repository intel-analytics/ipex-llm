#!/bin/bash
SGX=1 ./pal_loader bash -c "python ./work/examples/helloworld.py" | tee test-helloworld-sgx.log
