## 1. Build container image
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

If image is ready, you can run the container and enroll by using `run-docker-container.sh` in order to get a appid and apikey pair like below:

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
    -e EHSM_KMS_IP=$EHSM_KMS_IP \ # optional
    -e EHSM_KMS_PORT=$EHSM_KMS_PORT \ # optional
    -e KMS_TYPE=$KMS_TYPE \
    -e PCCS_URL=$PCCS_URL
    $ENROLL_IMAGE_NAME bash
    


```
## 3. enroll, generate key, encrypt and decrypt
```
# Enroll
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh enroll"

INFO [main.cpp(46) -> main]: ehsm-kms enroll app start.
INFO [main.cpp(86) -> main]: First handle:  send msg0 and get msg1.
INFO [main.cpp(99) -> main]: First handle success.
INFO [main.cpp(101) -> main]: Second handle:  send msg2 and get msg3.
INFO [main.cpp(118) -> main]: Second handle success.
INFO [main.cpp(120) -> main]: Third handle:  send att_result_msg and get ciphertext of the APP ID and API Key.

appid: d792478c-f590-4073-8ed6-2d15e714da78

apikey: bSMN3dAQGEwgx297Ff1H2umBzwzv6W34

INFO [main.cpp(155) -> main]: decrypt APP ID and API Key success.
INFO [main.cpp(156) -> main]: Third handle success.
INFO [main.cpp(159) -> main]: ehsm-kms enroll app end.


export appid=your_appid
export apikey=your_apikey
export container_input_file_path=mounted_address_of_host_input_file_path
export container_input_folder_path=mounted_address_of_host_input_folder_path


# Generatekeys
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh generatekeys $appid $apikey"

# Encrypt a single data file
# encrpted data is next to $container_input_file_path
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $container_input_file_path"


# Decrypt a single data file
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $apikey $container_input_file_path"

# SplitAndEncrypt
# encrpted data is in a directory next to $container_input_folder_path
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encryptwithrepartition $appid $apikey $container_input_folder_path"
```
## 4. Stop container:
```
docker stop $ENROLL_CONTAINER_NAME
```

