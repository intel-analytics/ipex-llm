export KMS_TYPE=azure
export ENROLL_IMAGE_NAME=your_enroll_image_name_built
export ENROLL_CONTAINER_NAME=your_enroll_container_name_to_run

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
    -e KMS_TYPE=$KMS_TYPE \
    $ENROLL_IMAGE_NAME bash


# 2. enroll
export KEY_VAULT_NAME=your_key_vault_name
export OBJECT_ID=your_system_identity

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh enroll $KEY_VAULT_NAME $OBJECT_ID"


# 3. generate primary key and data key, save them to local paths
export KEY_VAULT_NAME=your_key_vault_name

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh generatekeys $KEY_VAULT_NAME" # keys to save in $local_key_folder_path

# 4. encrypt and decrypt binary file
export DATA_FILE_NAME_TO_HANDLE=your_local_data_file_name_in_local_data_folder_path
export KEY_VAULT_NAME=your_key_vault_name

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh localcrypto $KEY_VAULT_NAME /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, a .encrypted and a .decrypted file will be generated


# 5. SplitAndEncrypt a CSV data file
export DATA_FILE_NAME_TO_HANDLE=your_local_csv_data_file_name_in_local_data_folder_path
export KEY_VAULT_NAME=your_key_vault_name

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh splitandencrypt $KEY_VAULT_NAME /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, a folder filled with encrypted data splits will be generated


# 6. Decrypt a encrypted CSV data file with SimpleQuery
export DATA_FILE_NAME_TO_HANDLE=your_local_encrypted_csv_data_file_name_in_local_data_folder_path
export KEY_VAULT_NAME=your_key_vault_name

sudo docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh simplequery $KEY_VAULT_NAME /home/data/$DATA_FILE_NAME_TO_HANDLE" # at local host side, decrypted files will be generated
