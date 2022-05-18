#!/bin/bash

base64result=$( openssl genrsa -3 3072 | base64 -w 0 )
sed -i "s@GENERATE_YOUR_ENCLAVE_KEY@$base64result@g" enclave-key-secret.yaml
kubectl apply -f enclave-key-secret.yaml
