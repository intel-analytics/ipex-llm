#!/bin/bash

$VAULT_NAME=$1

key=$(openssl genrsa -3 3072)

az keyvault secret set --vault-name $VAULT_NAME --name "enclave-key-pem" --value $key
