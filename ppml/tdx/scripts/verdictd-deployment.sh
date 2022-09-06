#!/bin/bash

set -e
set -x

git clone https://github.com/inclavare-containers/verdictd.git

curl -L -o opa https://openpolicyagent.org/downloads/v0.30.1/opa_linux_amd64_static
chmod 755 ./opa
mv opa /usr/local/bin/opa

apt install -y protobuf-compiler
cargo install bindgen

# Linux(Ubuntu)
apt-get install -y llvm-dev libclang-dev clang

make && make install
