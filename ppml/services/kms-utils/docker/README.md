## 1. Pull/Build container image

Download image as below:

```bash
docker pull intelanalytics/kms-utils:0.3.0-SNAPSHOT
```

Or you are allowed to build the image manually:
```
# set the arguments inside the build script first
bash build-docker-image.sh
```

## 2. Run container

This is the file structure we expect:
```
Folder --> set as host_data_folder_path when creating container
|
│
└───folder1 --> The corresponding mounted address will be set as container_input_folder_path
│       file11.txt --> Data file to be encrpted or decrypted, and the corresponding mounted address set as container_input_file_path
|
└───folder2
        file21.txt
```

If image is ready, you can run the container and enroll like below in order to get a appid and apikey pair like below:

```bash
export KMS_TYPE=an_optional_kms_type # KMS_TYPE can be (1) ehsm, (2) simple
export EHSM_KMS_IP=your_ehsm_kms_ip # if ehsm
export EHSM_KMS_PORT=your_ehsm_kms_port # if ehsm
export ENROLL_IMAGE_NAME=your_enroll_image_name_built
export ENROLL_CONTAINER_NAME=your_enroll_container_name_to_run
export PCCS_URL=your_pccs_url # format like https://x.x.x.x:xxxx/sgx/certification/v3/

sudo docker run -itd \
    --privileged \
    --net=host \
    --name=$ENROLL_CONTAINER_NAME \
    -v /dev/sgx/enclave:/dev/sgx/enclave \
    -v /dev/sgx/provision:/dev/sgx/provision \
    -v $host_data_folder_path:/home/data \
    -v $host_key_folder_path:/home/key \
    -e EHSM_KMS_IP=$EHSM_KMS_IP \
    -e EHSM_KMS_PORT=$EHSM_KMS_PORT \
    -e KMS_TYPE=$KMS_TYPE \
    -e PCCS_URL=$PCCS_URL \
    $ENROLL_IMAGE_NAME bash
```
## 3. Key Management and Data Crypto APIs

1. **Enroll:**

   ```bash
   curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"
   ```

   You are expected to receive response like:

   ```bash
   {"code":200,"message":"successful","result":{"apikey":"E8QKp******Sg","appid":"8d5dd******dcb8265981ba"}}
   ```

   Then set `appid`, `apikey` and other environment variables:

   ```bash
   export appid=your_appid
   export apikey=your_apikey
   export container_input_file_path=mounted_address_of_host_input_file_path
   export container_input_folder_path=mounted_address_of_host_input_folder_path
   export data_source_type=your_data_source_type # The data_source_type can be any one of csv, json, parquet, or textfile.
   # If you do not specify data_source_type, it will be set to csv by default
   ```

2.  **Generatekeys:**

    ```bash
    docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh generatekeys $appid $apikey"
    ```

3. **Encrypt File:**

   ```bash
   # encrpted data is next to $container_input_file_path
   docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $container_input_file_path" $data_source_type
   ```

4. **Decrypt File:**

   ```bash
   docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $apikey $container_input_file_path" $data_source_type
   ```

5. **Split a Large File and Encrypt:**

   ```bash
   # encrpted data is in a directory next to $container_input_folder_path
   docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encryptwithrepartition $appid $apikey $container_input_folder_path"
   ```

## 4. Stop container:
```bash
docker stop $ENROLL_CONTAINER_NAME
```
