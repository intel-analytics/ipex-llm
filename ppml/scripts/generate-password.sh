#!/bin/bash

mkdir -p password && cd password
export PASSWORD=$1 #used_password_when_generate_keys
openssl genrsa -out key.txt 2048
echo $PASSWORD | openssl rsautl -inkey key.txt -encrypt >output.bin
