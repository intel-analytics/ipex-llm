export KMS_TYPE=an_optional_kms_type # KMS_TYPE can be (1) ehsm, (2) simple
export EHSM_KMS_IP=your_ehsm_kms_ip # if ehsm
export EHSM_KMS_PORT=your_ehsm_kms_port # if ehsm
export ENROLL_IMAGE_NAME=your_enroll_image_name_built
export ENROLL_CONTAINER_NAME=your_enroll_container_name_to_run
export PCCS_URL=your_pccs_url # format like https://x.x.x.x:xxxx/sgx/certification/v3/

export local_data_folder_path=your_data_folder_path_to_create_at_local_host
export local_key_folder_path=your_key_folder_path_to_create_at_local_host

sudo mkdir $local_data_folder_path
sudo mkdir $local_key_folder_path

# 1. start container
sudo docker run -itd \
    --privileged \
    --net=host \
    --name=$ENROLL_CONTAINER_NAME \
    -v /dev/sgx/enclave:/dev/sgx/enclave \
    -v /dev/sgx/provision:/dev/sgx/provision \
    -v $local_data_folder_path:/home/data \
    -v $local_key_folder_path:/home/key \
    -e EHSM_KMS_IP=$EHSM_KMS_IP \
    -e EHSM_KMS_PORT=$EHSM_KMS_PORT \
    -e KMS_TYPE=$KMS_TYPE \
    -e PCCS_URL=$PCCS_URL
    $ENROLL_IMAGE_NAME bash


# 2. enroll
sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh enroll"


# 3. generate primary key and data key, save them to local paths
export APPID=your_appid_obtained_from_enroll
export APIKEY=your_apikey_obtained_from_enroll

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh generatekeys $APPID $APIKEY" # keys to save in $local_key_folder_path

# 4. encrypt and decrypt binary file
export DATA_FILE_NAME_TO_HANDLE=your_local_data_file_name_in_local_data_folder_path
export APPID=your_appid_obtained_from_enroll
export APIKEY=your_apikey_obtained_from_enroll

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh localcrypto $APPID $APIKEY /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, a .encrypted and a .decrypted file will be generated


# 5. SplitAndEncrypt a CSV data file
export DATA_FILE_NAME_TO_HANDLE=your_local_csv_data_file_name_in_local_data_folder_path
export APPID=your_appid_obtained_from_enroll
export APIKEY=your_apikey_obtained_from_enroll

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh splitandencrypt $APPID $APIKEY /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, a folder filled with encrypted data splits will be generated


# 6. Decrypt a encrypted CSV data file with SimpleQuery
export DATA_FILE_NAME_TO_HANDLE=your_local_encrypted_csv_data_file_name_in_local_data_folder_path
export APPID=your_appid_obtained_from_enroll
export APIKEY=your_apikey_obtained_from_enroll

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh simplequery $APPID $APIKEY /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, decrypted files will be generated
