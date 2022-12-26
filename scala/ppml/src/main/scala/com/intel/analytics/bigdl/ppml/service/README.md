# BigDL Remote Attestation Service

Attestation Service for SGX/TDX

## Requirements
Ubuntu 20.04
Intel SGX SDK
Intel DCAP packages
PCCS
BigDL

## Usage
```bash
java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.service.BigDLRemoteAttestationService -u <serviceURL> -p <servicePort> -s <httpsKeyStoreToken> -t <httpsKeyStorePath> -h <httpsEnabled>
```

## How to deploy a BigDL Remote Attestation Service
You can install all the required libs (Intel SGX SDK, DCAP, PCCS, BigDL, ... ) by your own, or you can refer [this]() to build a docker image and deploy the Attestation Service in a docker container.

### http service
After installation, start your server with command:
```bash
java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.service.BigDLRemoteAttestationService -u <serviceURL> -p <servicePort>
```
You will find ths console output like:
```bash
Server online at http://localhost:8184/
Press RETURN to stop...
```
which indicates the service is listening on `http://localhost:8184/` (default settings for example), and you can post a verify quote request to the URL.

### https service
For https, you need to generate a PKCS12 certificate.

```bash
# Generate all files in a temporary directory
mkdir key && cd key
# 1. Generete private key for server
openssl genrsa -des3 -out server.key 2048
# 2. Generate ca.crt
openssl req -new -x509 -key server.key -out ca.crt -days 3650
# 3. Generate Certificate Signing Request（CSR）
openssl req -new -key server.key -out server.csr
# 4. Generate certificate of server
openssl x509 -req -days 3650 -in server.csr -CA ca.crt -CAkey server.key -CAcreateserial -out server.crt
# 5. Merge server certificate
cat server.key server.crt > server.pem
# 6. Generate PKCS12 certificate
openssl pkcs12 -export -clcerts -in server.crt -inkey server.key -out server.p12 
```

Then you can start your server with command:
```bash
java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.service.BigDLRemoteAttestationService -u <serviceURL> -p <servicePort>
-s true -h key/server.p12 -t <your_token_of_server.key>
```