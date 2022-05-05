#!/bin/bash
mkdir -p password && cd password

export VAULT_NAME=$1
export PASSWORD=$2 #used_password_when_generate_keys

openssl genrsa -out key.txt 2048
echo $PASSWORD | openssl rsautl -inkey key.txt -encrypt >output.bin

az keyvault secret set --vault-name $VAULT_NAME --name "key-txt" --value $(base64 -w 0 key.txt)

az keyvault secret set --vault-name $VAULT_NAME --name "output-bin" --value $(base64 -w 0 output.bin)
