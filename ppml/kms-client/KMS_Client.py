import argparse
import FileOperator, KeyManager

def generate_primary_key(ip, port):
    KeyManager.generate_primary_key_ciphertext(ip, port)


def generate_data_key(ip, port, encrypted_primary_key_path):
    KeyManager.generate_data_key_ciphertext(ip, port, encrypted_primary_key_path)


def encrypt_file_without_key(data_file_path, ip, port):
    KeyManager.generate_primary_key_ciphertext(ip, port)
    KeyManager.generate_data_key_ciphertext(ip, port, './encrypted_primary_key')
    FileOperator.encrypt_data_file(ip, port, data_file_path, './encrypted_primary_key', './encrypted_data_key')


def encrypt_file_with_key(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    FileOperator.encrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path)


def decrypt_file(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    FileOperator.decrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path)


def encrypt_directory_with_key(dir_path, ip, port,encrypted_primary_key_path, encrypted_data_key_path, save_dir=None):
    FileOperator.encrypt_directory_automation(ip, port, dir_path, encrypted_primary_key_path, encrypted_data_key_path,save_dir)


def encrypt_directory_without_key(dir_path, ip, port, save_dir=None):
    KeyManager.generate_primary_key_ciphertext(ip, port)
    KeyManager.generate_data_key_ciphertext(ip, port, './encrypted_primary_key')
    FileOperator.encrypt_directory_automation(ip, port, dir_path, './encrypted_primary_key', './encrypted_data_key',save_dir)


def get_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    data_key_plaintext = KeyManager.retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    print(data_key_plaintext)
    return data_key_plaintext


def decrypt_csv_columns(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir):
    FileOperator.decrypt_csv_columns_automation(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-api', '--api', type=str, help='name of the API to use', required=True)
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-pkp', '--pkp', type=str, help='path of the primary key storage file', required=False)
    parser.add_argument('-dkp', '--dkp', type=str, help='path of the data key storage file', required=False)
    parser.add_argument('-dfp', '--dfp', type=str, help='path of the data file to encrypt', required=False)
    parser.add_argument('-dir', '--dir', type=str, help='path of the directory containing column-encrypted CSVs or the directory to be encrypted', required=False)
    parser.add_argument('-sdp', '--sdp', type=str, help='path of the save directory output to',required=False)
    args = parser.parse_args()

    api = args.api
    ip = args.ip
    port = args.port

    if api == 'encrypt_file_without_key':
        data_file_path = args.dfp
        encrypt_file_without_key(data_file_path, ip, port)
    elif api == 'generate_primary_key':
        generate_primary_key(ip, port)
    elif api == 'generate_data_key':
        encrypted_primary_key_path = args.pkp
        generate_data_key(ip, port, encrypted_primary_key_path)
    elif api == 'encrypt_file_with_key':
        data_file_path = args.dfp
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        encrypt_file_with_key(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    elif api == 'decrypt_file':
        data_file_path = args.dfp
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        decrypt_file(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    elif api == 'encrypt_directory_without_key':
        dir_path = args.dir
        save_path = args.sdp
        encrypt_directory(dir_path, ip, port, save_path)
    elif api == 'encrypt_directory_with_key':
        dir_path = args.dir
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        save_path = args.sdp
        encrypt_directory_with_key(dir_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path,save_path)
    elif api == 'get_data_key_plaintext':
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        get_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    elif api == 'decrypt_csv_columns':
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        input_dir = args.dir
        decrypt_csv_columns(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir)
