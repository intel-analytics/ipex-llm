#!/bin/bash

$VAULT_NAME=$1

base64result=$( openssl genrsa -3 3072 | base64 -w 0 )

az keyvault secret set --vault-name $VAULT_NAME --name "enclave-key-pem" --value $base64result
