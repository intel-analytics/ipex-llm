# BigDL Remote Attestation Service

Attestation Service for SGX/TDX

## Requirements
Ubuntu 20.04
Intel SGX SDK
Intel DCAP packages
PCCS

## Usage
```bash
java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.service.BigDLRemoteAttestationService -u <serviceURL> -p <servicePort> -s <httpsKeyStoreToken> -t <httpsKeyStorePath> -h <httpsEnabled>
```
