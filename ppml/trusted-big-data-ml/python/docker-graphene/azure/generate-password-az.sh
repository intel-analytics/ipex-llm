#!/bin/bash
export VAULT_NAME=$1
export PASSWORD=$2 #used_password_when_generate_keys

az keyvault secret set --vault-name $VAULT_NAME --name "key-pass" --value $PASSWORD

