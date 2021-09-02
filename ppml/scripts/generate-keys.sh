#!/bin/bash
OUTPUT=keys.yaml

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

echo apiVersion: v1 > $OUTPUT
echo data: >> $OUTPUT

echo " keystore.jks:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 keystore.jks  >> $OUTPUT
echo "" >> $OUTPUT

echo " keystore.pkcs12:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 keystore.pkcs12  >> $OUTPUT
echo "" >> $OUTPUT

echo " server.pem:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 server.pem  >> $OUTPUT
echo "" >> $OUTPUT

echo " server.crt:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 server.crt  >> $OUTPUT
echo "" >> $OUTPUT

echo " server.csr:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 server.csr  >> $OUTPUT
echo "" >> $OUTPUT

echo " server.key:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 server.key  >> $OUTPUT
echo "" >> $OUTPUT

echo kind: Secret  >> $OUTPUT
echo metadata:  >> $OUTPUT
echo " name: ssl-keys"  >> $OUTPUT
echo type: Opaque   >> $OUTPUT
