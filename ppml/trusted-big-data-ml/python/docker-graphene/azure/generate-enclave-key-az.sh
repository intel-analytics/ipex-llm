#!/bin/bash

VAULT_NAME=$1

openssl genrsa -3 -out /graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem 3072

az keyvault secret set --vault-name $VAULT_NAME --name "enclave-key-pem" --value "$(cat /graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem)"
