# EHSM KMS Client API

BigDL uses [eHSM](https://github.com/intel/ehsm/) (SGX Enclave Based Hardware Security Module) as one type of Key Management. The python EHSM KMS client APIs here enable users to communicate with a key management service, which allow them to generate and apply keys safely to encrypt/decrypt data files end-to-end.

## Requirement
```
requests
argparse
cryptography==3.3.2
```

## EHSM KMS Client API Examples

Here we list some basic usage examples of EHSM KMS Client, you can refer to `client.py` for details and more APIs as you need.

Please make sure EHSM KMS service has been [deployed](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#deploy-bigdl-ehsm-kms-on-kubernetes). Parameter `port` in below commands is optional as it is set to `3000`, which is the default value of EHSM KMS service.

First, [enroll](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#3-enroll-through-restapi) and export *APPID* and *APIKEY* as environment variables:

```
export APPID=YOUR_APPID
export APIKEY=YOUR_APIKEY
```

As they are the user certificates for KMS, please export them only in trusted and safe environment e.g. your host machine or a local docker container. For kubernetes users, we recommend to upload them to pods as kubernetes secrets.
 
- Request a primary key and save it locally in ciphertext:

  ```bash
  python client.py -api generate_primary_key -ip <KMS_IP> [-port <KMS_PORT>]
  ```

- Request a data key with prepared primary key and save it locally in ciphertext:

  ```bash
  python client.py -api generate_data_key -ip <KMS_IP> [-port <KMS_PORT>] -pkp <PRIMARYED_KEY_PATH> [-dkl <DATA_KEY_LENGTH_16_OR_32>]
  ```

- Encrypt a file without holding keys (keys will be generated automatically):

  ```bash
  python client.py -api encrypt_file_without_key -ip <KMS_IP> [-port <KMS_PORT>] -dfp <DATA_FILE_PATH>
  ```

- Encrypt a file with specified keys (if you have generated keys before, you will find encrypted key files named `encrypted_primary_key` and `encyrpted_data_key`, and you can use them again in the following):

  ```bash
  python client.py -api encrypt_file_with_key -ip <KMS_IP> [-port <KMS_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dfp <DATA_FILE_PATH>
  ```

- Decrypted a file with according keys:

  ```bash
  python client.py -api decrypt_file -ip <KMS_IP> [-port <KMS_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dfp <ENCYRPTED_DATA_FILE_PATH>
  ```

- Automatically generate keys and encrypt all files under a directory:

  ```bash
  python client.py -api encrypt_directory_without_key -ip <KMS_IP> [-port <KMS_PORT>] -dir <DIRECTORY_TO_BE_ENCRYPTED> [-sdp <SAVE_DIRECTORY_PATH>]
  ```
- Encrypt all files under a directory with existing keys:

  ```bash
  python client.py -api encrypt_directory_with_key -ip <KMS_IP> [-port <KMS_PORT>] -dir <DIRECTORY_TO_BE_ENCRYPTED> -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> [-sdp <SAVE_DIRECTORY_PATH>]

- Get plaintext data key from a `encyrpted_data_key`:

  ```bash
  python client.py -api get_data_key_plaintext -ip <KMS_IP> [-port <KMS_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH>
  ```

- Decrypt columns of CSVs under a directory:

  ```bash
  python client.py -api decrypt_csv_columns -ip <KMS_IP> [-port <KMS_PORT>] -pkp <PRIMARYED_KEY_PATH> -dkp <DATA_KEY_PATH> -dir <DIRECTORY_TO_BE_ENCRYPTED>
  ```
