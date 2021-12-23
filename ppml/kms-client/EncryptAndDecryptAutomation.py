import GeneratePrimaryKey,GenerateDataKey,EncryptFile,DecryptFile
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-dfp', '--dfp', type=str, help='path of the data file to encrypt', required=True)
    args = parser.parse_args()
    ip = args.ip
    port = args.port
    dfp = args.dfp
    return ip, port, dfp

if __name__ == "__main__":
    ip, port, dfp = get_args()
    GeneratePrimaryKey.generate_primary_key(ip, port)
    print('[INFO] Generate Primary Key Finished.')
    GenerateDataKey.generate_data_key(ip, port, './encrypted_primary_key')
    print('[INFO] Generate Data Key Finished.')
    EncryptFile.encrypt_file(ip, port, './encrypted_primary_key', './encrypted_data_key', dfp)
    print('[INFO]  Encrypt File Finished.')
    DecryptFile.decrypt_file(ip, port, './encrypted_primary_key', './encrypted_data_key', dfp+'.encrypted')
    print('[INFO] Decrypt File Finished.')
