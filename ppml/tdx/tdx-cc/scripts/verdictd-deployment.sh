#!/bin/bash

set -e
set -x

# Install dependencies
curl -L -o opa https://openpolicyagent.org/downloads/v0.30.1/opa_linux_amd64_static
chmod 755 ./opa
mv opa /usr/local/bin/opa
 
# Install bindgen
cargo install bindgen

# Install protobuf
# apt install -y protobuf-compiler # Ubuntu
yum install -y protobuf_codegen # Centos

# Install clang-libs clang-devel
# apt-get install -y llvm-dev libclang-dev clang # Ubuntu
yum install -y clang-libs clang-devel # Centos

# Build Verdictd
git clone -b 2022-poc https://github.com/jialez0/verdictd
cd verdictd
make && make install
cd -
