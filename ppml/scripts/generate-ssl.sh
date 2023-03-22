#!/bin/bash

mkdir -p /ppml/keys && cd /ppml/keys
# generate RAS keys
openssl genrsa -des3 -out server.key 2048
# generate a certificate signing request
openssl req -new -key server.key -out server.csr
# generate a self-signed X.509 certificate
openssl x509 -req -days 9999 -in server.csr -signkey server.key -out server.crt
