#!/bin/bash

PCCS_PORT=$PCCS_PORT
API_KEY=$API_KEY
HTTPS_PROXY_URL=$HTTPS_PROXY_URL

# Step 1. Generate certificates to use with PCCS
mkdir /opt/intel/pccs/ssl_key
cd /opt/intel/pccs/ssl_key
openssl genrsa -out private.pem 2048
openssl req -new -key private.pem -out csr.pem \
        -subj "/C=$COUNTRY_NAME/ST=$CITY_NAME/L=$CITY_NAME/O=$ORGANIZATION_NAME/OU=$ORGANIZATION_NAME/CN=$COMMON_NAME/emailAddress=$EMAIL_ADDRESS/" -passout pass:$PASSWORD -passout pass:$PASSWORD
openssl x509 -req -days 365 -in csr.pem -signkey private.pem -out file.crt
rm -rf csr.pem
chmod 644 ../ssl_key/*
ls ../ssl_key

# Step 2. Set default.json to be under ssl_key folder and fill the parameters
cd /opt/intel/pccs/config/

userTokenHash=$(echo -n "user_password" | sha512sum | tr -d '[:space:]-')
adminTokenHash=$(echo -n "admin_password" | sha512sum | tr -d '[:space:]-')
HOST_IP=0.0.0.0

sed -i "s/YOUR_HTTPS_PORT/$PCCS_PORT/g" default.json
sed -i "s/YOUR_HOST_IP/$HOST_IP/g" default.json
sed -i 's@YOUR_PROXY@'"$HTTPS_PROXY_URL"'@' default.json
sed -i "s/YOUR_USER_TOKEN_HASH/$userTokenHash/g" default.json
sed -i "s/YOUR_ADMIN_TOKEN_HASH/$adminTokenHash/g" default.json
sed -i "s/YOUR_API_KEY/$API_KEY/g" default.json
chmod 644 default.json
cd /opt/intel/pccs/

# Step 3. Start PCCS service and keep listening
/usr/bin/node -r esm pccs_server.js
