#!/bin/bash

mkdir -p /ppml/keys && cd /ppml/keys
# generate RAS keys using password
openssl genrsa -des3 -out server.key 2048
# generate a certificate signing request
openssl req -new -key server.key -out server.csr
# generate a self-signed X.509 certificate
openssl x509 -req -days 9999 -in server.csr -signkey server.key -out server.crt

# get server.key and server.crt without password
cat server.key | sudo tee server.pem
cat server.crt | sudo tee -a server.pem
openssl rsa -in server.pem -out server.key
openssl x509 -in server.pem -out server.crt
