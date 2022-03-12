## **KMS Scala Start-up with LocalCryptoExample**

## 1. Prerequisites

- Ensure you have a running [eHSM server](https://github.com/intel-analytics/BigDL/tree/main/ppml/kms-client#1-deploy-bigdl-ppml-with-kms-server).
- Ensure you have a [BigDL PPML container](https://github.com/intel-analytics/BigDL/tree/main/ppml/kms-client#31-start-a-bigdl-ppml-container-with-kms-client) as data processing environment.

## 2. En/decrypt a data file with SimpleKeyManagementService

SimpleKeyManagementService is implemented on a in-memory enroll mechanism and simple key generation, which does not rely on ehsm server, and can be used to test your workflow.

Enter the BigDL PPML container:

```bash
docker exec -it kms-client bash
cd /ppml/trusted-big-data-ml
```

Prepare a single data file to process, and submit the en/decrypt job to spark by the following:

```bash
bash work/kms-client/scala/simple-local-cryptos-example.sh <your_data_file_path> <spark_local_ip>
```

Then, you will find primary key, data key at `./encrypted_primary_key`, `./encrypted_data_key` respectively. Also, results are saved at `<your_data_file_path>.encrypted`,  `<your_data_file_path>.decrypted`, which are encrypted ciphertext and decrypted plaintext respectively.

If you want to run the spark job in SGX mode, do as the following:

```bash
bash init.sh
bash work/kms-client/scala/simple-local-cryptos-example-sgx.sh <your_data_file_path> <spark_local_ip>
```

## 3. En/decrypt a data file with EHSMKeyManagementService

Enter the BigDL PPML container:

```bash
docker exec -it kms-client bash
cd /ppml/trusted-big-data-ml
```

Prepare a single data file to process, and submit the en/decrypt job to spark by the following:

```bash
bash work/kms-client/scala/ehsm-local-cryptos-example.sh <your_data_file_path> <ehsm_kms_server_ip> <ehsm_kms_server_port> <spark_local_ip>
```

Then, you will find primary key, data key at `./encrypted_primary_key`, `./encrypted_data_key` respectively. Also, results are saved at `<your_data_file_path>.encrypted`,  `<your_data_file_path>.decrypted`, which are encrypted ciphertext and decrypted plaintext respectively.
If you want to run the spark job in SGX mode, do as the following:
```bash
bash init.sh
bash work/kms-client/scala/ehsm-local-cryptos-example-sgx.sh <your_data_file_path> <spark_local_ip>
```
