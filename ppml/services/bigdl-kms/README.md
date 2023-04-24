# BigDL KMS (Key Management Service) on Kubernetes

- [Architecture](#architecture)
- [Threat Model](#threat-model)
- [Start on Kubernetes](#start-on-kubernetes)
  - [1. Pre-requests](#1-pre-requests)
  - [2. Prepare SSL Key and Password](#2-prepare-ssl-key-and-password)
  - [3. Deploy Service](#3-deploy-service)
  - [4. Check Deployment](#4-check-deployment)
- [REST APIs](#rest-apis)
  - [1. Test Service Readiness](#1-test-service-readiness)
  - [2. Enroll](#2-enroll)
  - [3. Generate Primary Key](#3-generate-primary-key)
  - [4. Generate Data Key from the Primary Key](#4-generate-data-key-from-the-primary-key)
  - [5. Retrieve Plain Text of the Data Key](#5-retrieve-plain-text-of-the-data-key)
- [Stop Service](#stop-service)

## Architecture
![BigDLKMS](https://user-images.githubusercontent.com/60865256/229735029-b93f221a-7973-49fa-9474-a216121caf18.png)

It highlights that a multi-level wrapping of encryption keys is implemented to mitigate the risk of wrapping key leakage.

1. The *rootK*, a global key which encrypts and decrypts all of *userK*s (keys generated as request to operate application data). Due to its importance, we provide detailed [generation and protection](#root-key-generation-and-protection) towards it.
2. An user send a request from his application to the KMS server in order to retrieve a *UserK*, e.g. primary key or data key.
3. KMS server generates the requested-specify key, encrypts the generation with *rootK*, and saves the insensitive ciphertext (and response the user if needed). The whole process above is executed in TEE-protected memory region.
4. When receiving retrieve request of plain-text data key, the server queries, decrypts and responses with the target, which is TEE-protected as well.

The encrypted keys are saved in persistent storage mounted as kubernetes volume, and this achieves both privacy preservation and fault tolerance.

## Threat Model

BigDL KMS protects against the following potential attacks:

1. Secretly listen to any communication among KMS clients, server and storage.
2. Access to any data in transit (over network) or at rest (server backend e.g. cloud storage).
3. Access to unauthenticated data belong to the other parties.
4. Hack to memory or runtime of the KMS server.

BigDL KMS uses multiple security techniques e.g. Trusted Execution Environment (SGX/TDX), AES Crypto, TLS/SSL and Remote Attestation etc. to ensure end-to-end secure key management even in an untrusted environment with above various parts.

## Root Key Generation and Protection

The root key encrypts and decrypts other keys of user, and application data of user will be hacked or corrupted once it is leaked or lost. Therefore, root key should be handled securely and in fault-tolerance. 

BigDL KMS is able to create root key in two ways, which are trade-off between deploy-ability, fault-tolerance, security and privacy-preservation.

### 1. Root Key as Kubernetes Secret

User creates at client and uploads the root key as a [kubernetes secret](https://kubernetes.io/docs/concepts/configuration/secret/):

This require user to directly specify a root key string when [deploying the service in the following section](#3-deploy-service).

### 2. Root Key with Secret Sharing

User calls [rotate API firstly, which lets the KMS server to generate the root key in its safe memory. Next, the KMS server will apply [Shamir Secret Sharing](https://en.wikipedia.org/wiki/Secret_sharing) to split the root key into shares, and send them to end user. 

Here, the original root key will never goes of TEE. If the server crashed, user is allowed to call [recover][#5-initliz] API, and send 3/5 of the shares he received previously, and this will recover the original key at server.

## Start on Kubernetes

For users who want to deploy BigDL KMS as a lightweight micro-service, we also provide a [docker version](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/bigdl-kms/docker/run-docker-container.sh), and the parameters in the script are the same as the following.

### 1. Pre-requests

- Make sure you have a workable **Kubernetes cluster/machine**
- Prepare [base bigdl-kms docker image](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/easy-kms/docker#1-pullbuild-container-image). **Note** that if enable SGX, please build a custom-signed image by yourself, or pull our reference image (signed by open key of BigDL and do not use it in production).

### 2. Prepare SSL Key and Password

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#prepare-key-and-password) to generate keys and passwords for TLS encryption, and upload them to k8s as secrets.

### 3. Deploy Service

Go to `kubernetes` directory:

```bash
cd kubernetes
```

Modify parameters in script `install-bigdl-kms.sh`:

```
......
imageName       ---> if enable SGX, replace with custom image
dataStoragePath ---> a host path for persistent stoarge
serviceIP       ---> your key management service ip to expose
rootKey         ---> a user-provides 256bit base64 AES key string
teeMode         ---> Mode of TEE the server is going to run in, sgx or tdx
......
```

If you choose to [generate root key with secret sharing](#2-root-key-with-secret-sharing), please ignore the `rootKey` parameter by setting it to empty in the script:

```bash
...
export rootKey=""
...
```

Otherwise, you can choose to directly provide the `rootKey` as a [Kubernetes secret](#1-root-key-as-kubernetes-secret). Here, you can create a 256b-AES-base64 key string in this way:

```bash
openssl enc -aes-256-cbc -k secret -P -md sha1
# you will get a key, and copy it to below field
echo <key_generated_above> | base64
```

Then, deploy easy-kms on kubernetes by one command:

```bash
bash install-bigdl-kms.sh
```

### 4. Check Deployment

Check the service whether it has successfully been running (it may take seconds).
```bash
kubectl get all | grep bigdl-key-management
```

It is expected to get outputs similar to the following:
```bash
pod/bigdl-key-management-server-f4985d65f-2qctz    1/1            Running
service/bigdl-key-management-service               LoadBalancer   10.99.15.77     <serviceIP>    9875:31157/TCP
deployment.apps/bigdl-key-management-server        1/1            1               1
replicaset.apps/bigdl-key-management-server-f4985d65f
```

## REST APIs

You can communicate with BigDL KMS using client [BigDLKeyManagementService](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/kms/client/BigDLKeyManagementService.scala), or simply verify through requesting REST API like below. The default port number of bigdl kms is `9876`.

### 1. Rotate Root Key

Call `rotate` if you want to [initialize the root key with secret sharing](#2-root-key-with-secret-sharing), or update the key whenever you do not believe it any more:

```bash
curl -X GET -k -v "https://<kmsIP>:<kmsPort>/rootKey/rotate"
```

You will receive secret shares as below, and please save them including both index and content, since they are needed to recover the root key once the server crashed:

```bash
{{
  "index" : 1,
  "content" : "T*****U="
},{
  "index" : 2,
  "content" : "Q*****U="
},{
  "index" : 3,
  "content" : "R*****0="
},{
  "index" : 4,
  "content" : "R*****A="
},{
  "index" : 5,
  "content" : "Q*****g="
}}
```

### 2. Test Service Readiness

```bash
curl -k -v "https://<kmsIP>:<kmsPort>/" # default port of bigdl-kms is 9876 and can be configured in bigdl-kms.yaml
```

you will get similar to below：

```bash
welcome to BigDL-Key-Management-Service

create a user like: POST /user/{userName}?token=a_token_string_for_the_user

get a primary key like: POST /primaryKey/{primaryKeyName}?user=your_username&&token=your_token

get a data key like: POST /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token

get the data key like: GET /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token1

```

### 3. Enroll

```bash
curl -X POST -k -v "https://<kmsIP>:<kmsPort>/user/<userName>?token=<userToken>"
user [<userName>] is created successfully!
```

### 4. Generate Primary Key

```bash
curl -X POST -k -v "https://<kmsIP>:<kmsPort>/primaryKey/<primaryKeyName>?user=<userName>&&token=<userToken>"
primaryKey [<primaryKeyName>] is generated successfully!
```

### 5. Generate Data Key from the Primary Key

```bash
curl -X POST -k -v "https://<kmsIP>:<kmsPort>/dataKey/<dataKeyName>?user=<userName>&&token=<userToken>&&primaryKeyName=<primaryKeyName>"
dataKey [<dataKeyName>] is generated successfully!
```

### 6. Retrieve Plain Text of the Data Key

```bash
curl -X GET -k -v "https://<kmsIP>:<kmsPort>/dataKey/<dataKeyName>?user=<userName>&&token=<userToken>&&primaryKeyName=<primaryKeyName>"
XY********Yw==
```

### 7. Recover Root Key

When restarting the server, send secret shares received when calling [rotate](#1-rotate-root-key) one by one:

```bash
curl -X PUT -k -v "https://<kmsIP>:<kmsPort>/rootKey/recover" --data '{"index" : <index_of_a_secret>,"content" : "<content_of_a_secret>"}'
```

Repeat this command with 3 different secret shares, and the server will be recovered if right ones is received successfully.

## Stop Service

```
export teeMode=... # sgx or tdx
bash uninstall-bigdl-kms.sh
```


