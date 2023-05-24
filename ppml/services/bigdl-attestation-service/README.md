# BigDL Attestation Service

## Basic Attestation Workflow with BigDL Attestation Service
<img width="1049" alt="image" src="https://user-images.githubusercontent.com/109123695/234484380-a1b8325a-9114-488e-8746-600eccc994ac.png">

1. Admin register policy (requirements for attestation).
2. User launch application.
3. Application generate quote in TEE and send quote to Attestation Service.
4. Attestation Service verify the quote and respond with result.

## Prepare 
You need to prepare the directoryies of service data and keys. The `data` directory will save the service data encryptedly and the `keys` directory should place certificates for service. For https, you need to generate a PKCS12 certificate.

```bash
mkdir data keys 
# Generate all files in keys directory
cd keys
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

***You can also deploy BigDL Attestation Service without configuring `data` dir, but the service data may be lost when the docker container or kubenetes pod crashed.***

## Deploy on Kubernetes
```bash
cd ./kubernetes
# 1. Create namespace bigdl-remote-attestation-service if not exists
kubectl create namespace bigdl-remote-attestation-service

# 2. Configure bigdl-attestation-service-data, bigdl-attestation-service-keys, PCCS_URL, ATTESTATION_SERVICE_HOST and ATTESTATION_SERVICE_PORT in bigdl-attestation-service.yaml
vi bigdl-attestation-service.yaml

# 3. Apply bigdl-attestation-service.yaml
kubectl apply -f bigdl-attestation-service.yaml
```


## REST APIs

You can do attestation via REST API of BigDL Remote Attestation Service. Currently supported REST APIs are listed:

### Quick Check

* REST API format:
```
GET <bigdl_remote_attestation_address>/
```

### Enroll 

Obtain a valid access key pair (APPID and APIKey).

* REST API format:
```
GET <bigdl_remote_attestation_address>/enroll
```

* Response Data:

| Name | Type | Reference Value | Description |
|:-----------|:-----------|:-----------|:-----------|
| appID | String | HfpPKHdF... | ID which represents a certain application  |
| apiKey | String | EnJnmF31... | The application's access key to the BigDL Remote Attestation Service |

* Example Response:
```json
{
    "appID": "e95dfcd1-d98a-4b33-80ab-5249315ab5d4",
    "apiKey": "CHUIYoW0HF8b6Ig6q7MiOHHn2KGGQ3HL"
}
```

### Register Policy

Register a attestation policy for SGX/TDX quote, which will check some certain attributes or contents from the quote.

**TDX Policy is not implemented yet.**

* REST API format:
```
POST <bigdl_remote_attestation_address>/registerPolicy
```
* Request Payload:

| Name | Type | Reference Value | Description |
|:-----------|:-----------|:-----------|:-----------|
| appID | String | HfpPKHdF... | ID which represents a certain application  |
| apiKey | String | EnJnmF31... | The application's access key to the BigDL Remote Attestation Service |
| policyType | String | SGXMREnclavePolicy | Type of policy the service will check, currently support `SGXMREnclavePolicy` |
| mrEnclave | String | e38104fe6938... | (**For SGX**) The hash over the enclave pages loaded into the SGX protected memory |
| mrSigner | String | d412a4f07ef8... | (**For SGX**) The hash of the public portion of the key used to sign the enclave |

* Example Request:
```json
{
    "appID": "e95dfcd1-d98a-4b33-80ab-5249315ab5d4",
    "apiKey": "CHUIYoW0HF8b6Ig6q7MiOHHn2KGGQ3HL",
    "policyType": "SGXMREnclavePolicy",
    "mrEnclave": "2b0376b7ce6e2ece6279b2b49157841c..."
}
```
* Response Data:

| Name | Type | Reference Value | Description |
|:-----------|:-----------|:-----------|:-----------|
| policyID | string | 49ac44ff-01f0... | Generated policyID |

* Example Response:
```json
{
    "policyID": "49ac44ff-01f0-4f59-8d7e-fcffb78f0f4c"
}
```

### Verify SGX/TDX Quote

Verify a SGX/TDX quote with BigDL Remote Attestation Service.

* REST API format:
```
POST <bigdl_remote_attestation_address>/verifyQuote
```
* Request Payload:

| Name | Type | Reference Value | Description |
|:-----------|:-----------|:-----------|:-----------|
| appID | String | HfpPKHdF... | ID which represents a certain application  |
| apiKey | String | EnJnmF31... | The application's access key to the BigDL Remote Attestation Service |
| quote | String | AwACAAAAAAAJAA0Ak5py... | A valid DCAP quote in BASE64 |
| policyID | string | LdPXuPs8fI5Q... | (**Optional**) The policy which the given quote should satisfy |

* Example Request:
```json
{
    "quote": "AwACAAAAAAAJAA0Ak5pyM...",
    "appID":"e95dfcd1-d98a-4b33-80ab-5249315ab5d4",
    "apiKey":"CHUIYoW0HF8b6Ig6q7MiOHHn2KGGQ3HL",
    "policyID":"49ac44ff-01f0-4f59-8d7e-fcffb78f0f4c"
}
```
* Response Data:

| Name | Type | Reference Value | Description |
|:-----------|:-----------|:-----------|:-----------|
| result | Int | 0/1/-1 | 0 for success, 1 for warning, -1 for error |

* Example Response:
```json
{
    "result": 0
}
```
