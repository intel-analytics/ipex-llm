import argparse
import FileOperator, KeyManager

def encrypt_file_without_key(data_file_path, ip, port):
    KeyManager.generate_primary_key_ciphertext(ip, port)
    KeyManager.generate_data_key_ciphertext(ip, port, './encrypted_primary_key')
    FileOperator.encrypt_data_file(ip, port, data_file_path, './encrypted_primary_key', './encrypted_data_key')

def encrypt_file_with_key(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    FileOperator.encrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path)

def decrypt_file(data_file_path, ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    FileOperator.decrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path)

def encrypt_db_automation(db_path, ip, port):
    import shutil
    FileOperator.convert_db_to_csv(dp_path)
    KeyManager.generate_primary_key_ciphertext(ip, port)
    KeyManager.generate_data_key_ciphertext(ip, port, '.\encrypted_primary_key')
    encrypted_dir = db_path + '.encrypted'
    os.mkdir(encrypted_dir)
    FileOperator.encrypt_files_automation(ip, port, input_dir, encrypted_primary_key_path, encrypted_data_key_path)
    shutil.rmtree(db_path + '.csv')

def get_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    return KeyManager.retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)

def decrypt_csv_columns(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir):
    FileOperator.decrypt_csv_columns_automation(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir)

if __name__ == "__main__":
    parser.add_argument('-api', '--api', type=str, help='name of the API to use', required=True)
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-pkp', '--pkp', type=str, help='path of the primary key storage file', required=False)
    parser.add_argument('-dkp', '--dkp', type=str, help='path of the data key storage file', required=False)
    parser.add_argument('-dfp', '--dfp', type=str, help='path of the data file to encrypt', required=False)
    parser.add_argument('-dbp', '--dbp', type=str, help='path of the .db file to be processed', required=False)
    parser.add_argument('-dir', '--dir', type=str, help='path of the directory containing column-encrypted CSVs', required=False)
    args = parser.parse_args()
    
    api = args.api
    ip = args.ip
    port = args.port
    
    if api == 'encrypt_file_without_key':
        data_file_path = args.dfp
        encrypt_file_without_key(data_file_path, ip, port)
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
    elif api == 'encrypt_db_automation':
        db_path = agrs.dbp
        encrypt_db_automation(db_path, ip, port)
    elif api == 'get_data_key_plaintext':
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        get_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    elif api == 'decrypt_csv_columns':
        encrypted_primary_key_path = args.pkp
        encrypted_data_key_path = args.dkp
        input_dir = args.dir
        decrypt_csv_columns(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir)
