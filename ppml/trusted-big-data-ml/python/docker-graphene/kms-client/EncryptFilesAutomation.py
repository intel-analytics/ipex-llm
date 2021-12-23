import GeneratePrimaryKey,GenerateDataKey,EncryptFile
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-input_dir', '--input_dir', type=str, help='directory where input files to encrypt at', required=True)
    parser.add_argument('-save_dir', '--save_dir', type=str, help='directory where encrypted files are put', required=True)
    args = parser.parse_args()
    ip = args.ip
    port = args.port
    input_dir = args.input_dir
    save_dir = args.save_dir
    return ip, port, input_dir,save_dir

def encrypt_files(ip, port, pkp, dkp, input_dir, save_dir):
    for filename in os.listdir(input_dir):
        dfp = os.path.join(input_dir, filename)
        save_path = os.path.join(save_dir, filename + '.encrypted')
        EncryptFile.encrypt_file(ip, port, pkp, dkp, dfp, save_path)

if __name__ == "__main__":
    ip, port, input_dir, save_dir = get_args()
    print('[INFO] Generate Primary Key Start...')
    GeneratePrimaryKey.generate_primary_key(ip, port)
    print('[INFO] Generate Data Key Start...')
    GenerateDataKey.generate_data_key(ip, port, './encrypted_primary_key')
    print('[INFO]  Encrypt Files Start...')
    encrypt_files(ip, port, './encrypted_primary_key', './encrypted_data_key', input_dir, save_dir)
