#!/bin/bash

mkdir -p keys && cd keys
openssl genrsa -des3 -out server.key 2048
openssl req -new -key server.key -out server.csr
openssl x509 -req -days 9999 -in server.csr -signkey server.key -out server.crt
#cat server.key > server.pem
#cat server.crt >> server.pem

# Use sudo on cat if necessary.
# Changing from redirection to tee to elude permission denials.

cat server.key | sudo tee server.pem
cat server.crt | sudo tee -a server.pem
openssl pkcs12 -export -in server.pem -out keystore.pkcs12
keytool -importkeystore -srckeystore keystore.pkcs12 -destkeystore keystore.jks -srcstoretype PKCS12 -deststoretype JKS
openssl pkcs12 -in keystore.pkcs12 -nodes -out server.pem
openssl rsa -in server.pem -out server.key
openssl x509 -in server.pem -out server.crt
