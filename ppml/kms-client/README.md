# KMS Client Usage for BigDL PPML with KMS

BigDL implements its key management system based on [eHSM](https://github.com/intel/ehsm/) (SGX Enclave Based Hardware Security Module). It enables users to receive a central key management service by using client APIs here, which allow users to generate keys and encrypt/decrypt data files safely and conveniently.

In the following, we will present a end-to-end BigDL PPML with KMS example.

<div align="center">
<p align="center"> <img src="../../docs/readthedocs/image/KMS-Client.png" height="220px"><br></p>
KMS Client Architecture
</div>

## 1. Deploy BigDL PPML with KMS Server

The KMS server is based on open-sourced project [eHSM](https://github.com/intel/ehsm/) , it can be downloaded and deployed following commands below:

```
git clone https://github.com/intel/ehsm.git 
cd ehsm
./docker/build_and_run.sh -b
./docker/build_and_run.sh -r
```

Then, you will get a KMS server running in a docker container. If encounter any problem, you can refer to [eHSM](https://github.com/intel/ehsm/). In the future, this can be deployed as a K8S service from a deployment yaml.

## 2. Run BigDL PPML with KMS Client at Spark Local Mode

- ##### **Start a BigDL PPML Container with KMS Client**

  The KMS client in this example is running in the environment of [SGX-based Trusted Big Data ML](https://github.com/intel-analytics/BigDL/tree/branch-2.0/ppml/trusted-big-data-ml/python/docker-graphene), in which we have deployed necessary requirements. You can also do the same in other trusted execution environments, like Trusted Realtime ML etc., after installing python modules listed in `requirement.yml` with pip. 

  Prepare and start the client container as below:

  ```bash
  export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
  export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
  export LOCAL_IP=YOUR_LOCAL_IP
  export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14.0-SNAPSHOT
  sudo docker pull $DOCKER_IMAGE
  sudo docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="0-5" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
      --name=kms-client \
      -e LOCAL_IP=$LOCAL_IP \
      -e SGX_MEM_SIZE=64G \
      $DOCKER_IMAGE bash
  ```

  Pay attention to the variables and set them to correct values according to your operating environment.

- **Run The End-to-end BigDL PPML with KMS Example**

  A file encryption/decryption example is used to illustrate the KMS, in which the decryption process is done as a spark job in SGX. Its workflow can be seen as below:

  <div align="center">


  <p align="center"> <img src="../../docs/readthedocs/image/KMS_End-to-end_Example.png" height="300px"><br></p>

  </div>

  

  Enter the client container deployed in the previous step and run the below commands. Please modify `<KMS_SERVER_IP>` and `<INPUT_DATA_DIR_PATH>` according to your environment:

  ```bash
  docker exec -it kms-client bash
  cd /ppml/trusted-big-data-ml
  
  # You can also find the below shell script at /ppml/trusted-big-data-mlwork/kms-client/kms-e2e-example.sh
  
  export KMS_SERVER_IP=<KMS_SERVER_IP> # Ip Address Of The Server Host Deployed In The Previous Section
  export INPUT_DIR_PATH=<INPUT_DATA_DIR_PATH> # Path Of The Directory Containing CSVs
  
  # Step 1. Generate Primary Key And Data Key
  python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
    --api generate_primary_key --ip $KMS_SERVER_IP
  python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
    --api generate_data_key --ip $KMS_SERVER_IP --pkp /ppml/trusted-big-data-ml/encrypted_primary_key
  
  # Step 2. Encyypt The Input Directory Outside SGX With KMS
  python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
    --api encrypt_directory_with_key --ip $KMS_SERVER_IP  --dir $INPUT_DIR_PATH \
    --pkp /ppml/trusted-big-data-ml/encrypted_primary_key --dkp /ppml/trusted-big-data-ml/encrypted_data_key
  encrypted_path=$INPUT_DIR_PATH.encrypted
  
  # Step 3. Decrypt The Encrypted Files As A Spark Job Inside SGX And Then Encrypt Columns
  status_8_scala_e2e=1
  if [ $status_8_scala_e2e -ne 0 ]; then
  SGX=1 ./pal_loader bash -c "bash ./work/kms-client/DecryptFilesWithSpark.sh $encrypted_path $KMS_SERVER_IP $LOCAL_IP" 2>&1 > spark-inside-sgx.log
  fi
  status_8_scala_e2e=$(echo $?)
  output_path=$encrypted_path.col_encrypted
  
  # Step 4. Decrypt The colums And Ouput With KMS API
  python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py --api decrypt_csv_columns --ip 192.168.0.112 --dir $output_path --pkp /ppml/trusted-big-data-ml/encrypted_primary_key --dkp /ppml/trusted-big-data-ml/encrypted_data_key
  ```

  Then it takes a little time to operate in SGX, and you are expected to get output files under directory `$output_path`.

  You are also allowed to modify variables in `/ppml/trusted-big-data-mlwork/kms-client/kms-e2e-example.sh` and run this shell script. After that , you will get screen output similar to below:

  ```bash
  [INFO] Start To Generate Keys...
  [INFO] Primary Key Generated Successfully at ./encrypted_primary_key
  [INFO] Data Key Generated Successfully at ./encrypted_data_key
  [INFO] Start To Encrypt...
  [INFO] Encrypt Files Start...
  [INFO] Encrypt Successfully! Encrypted Output Is /ppml/trusted-big-data-ml/1m_csv.encrypted/tmp_mock_r_table.csv.encrypted
  ......
  [INFO] Encrypted Files.
  [INFO] The Directory Is Encrypted, Saved At /ppml/trusted-big-data-ml/1m_csv.encrypted
  [INFO] Decrypt The Ciphere Files Inside SGX...
  ......
  [INFO] The Output is Saved At /ppml/trusted-big-data-ml/1m_csv.encrypted.col_encrypted
  [INFO] Retrieve Output At Client Side.
  [INFO] Start To Decrypt Columns...
  [INFO] Column Decryption Start...
  [INFO] Decryption Finished. The Output Is /ppml/trusted-big-data-ml/1m_csv.encrypted.col_encrypted/tmp_mock_a_table.csv.encrypted/part-00001-7357c8d7-33bd-4173-b306-6fdbe5657591-c000.csv.col_decrypted
  ......
  [INFO] Total Elapsed Time For Columns Decrytion: 940.2687749862671 s
  ```

## 3. Run BigDL PPML with KMS Client at K8S Mode

This will be released in the future.

## 4. BigDL PPML with KMS Client API Usage

You can do below operations supported by BigDL PPML with KMS Client APIs. Please make sure KMS Server has been prepared. Parameter `port` in below commands is optional as it is set to `3000`, which is the default value of KMS Server.

- Request a primary key and save it locally in ciphertext:

  ```bash
  python ./work/kms-client/KMS_Client.py -api generate_primary_key -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>]
  ```

- Request a data key with prepared primary key and save it locally in ciphertext:

  ```bash
  python ./work/kms-client/KMS_Client.py -api generate_primary_key -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -pkp <PRIMARYED_KEY_PATH>
  ```

- Encrypt a file without holding keys (keys will be generated automatically):

  ```bash
  python ./work/kms-client/KMS_Client.py -api encrypt_file_without_key -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -dfp DATA_FILE_PATH
  ```

- Encrypt a file with specified keys (if you have generated keys before, you will find encrypted key files named `encrypted_primary_key` and `encyrpted_data_key`, and you can use them again in the following):

  ```bash
  python ./work/kms-client/KMS_Client.py -api encrypt_file_with_key -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dfp DATA_FILE_PATH
  ```

- Decrypted a file with according keys:

  ```bash
  python ./work/kms-client/KMS_Client.py -api decrypt_file -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dfp ENCYRPTED_DATA_FILE_PATH
  ```

- Encrypt all files under a directory:

  ```bash
  python ./work/kms-client/KMS_Client.py -api encrypt_directory -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -dir <DIRECTORY_TO_BE_ENCRYPTED>
  ```

- Get plaintext data key from a `encyrpted_data_key`:

  ```bash
  python ./work/kms-client/KMS_Client.py -api get_data_key_plaintext -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH>
  ```

- Decrypt columns of CSVs under a directory:

  ```bash
  python ./work/kms-client/KMS_Client.py -api decrypt_csv_columns -ip <kMS_SERVER_IP> [-port <KMS_SERVER_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dir <DIRECTORY_TO_BE_ENCRYPTED>
  ```

