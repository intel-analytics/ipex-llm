#!/bin/bash
cd /ppml/trusted-big-data-ml
/graphene/Tools/argv_serializer bash -c "python ./work/examples/helloworld.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash | tee test-helloworld-sgx.log && \
        cat test-helloworld-sgx.log | egrep -a "Hello"
