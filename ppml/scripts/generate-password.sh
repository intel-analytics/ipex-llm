#!/bin/bash

mkdir password && cd password
export YOUR_PASSWORD=$1 #used_password_in_generate-keys.sh
openssl genrsa -out key.txt 2048
echo $YOUR_PASSWORD | openssl rsautl -inkey key.txt -encrypt >output.bin
