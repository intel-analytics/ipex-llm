from KeyManager import retrieve_data_key_plaintext
from cryptography.fernet import Fernet
import os, sqlite3, csv

def read_data_file(data_file_path):
    with open(data_file_path, 'rb') as file:
        original = file.read()
    return original

def write_data_file(data_file_path, content):
    with open(data_file_path, 'wb') as file:
        file.write(content)

def encrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path, save_path=None):
    data_key = retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    fernet = Fernet(data_key)
    encrypted = fernet.encrypt(read_data_file(data_file_path))
    if save_path is None:
        save_path = data_file_path + '.encrypted'
    write_data_file(data_file_path, encrypted)
    print('[INFO] Encrypt Successfully! Encrypted Output Is ' + save_path)

def decrypt_data_file(ip, port, data_file_path, encrypted_primary_key_path, encrypted_data_key_path, save_path=None):
    data_key = retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    fernet = Fernet(data_key)
    decrypted = fernet.decrypt(read_data_file(data_file_path))
    if save_path is None:
        save_path = encrypted_file_path + '.decrypted'
    write_data_file(data_file_path, decrypted)
    print('[INFO] Decrypt Successfully! Decrypted Output Is ' + save_path)

def encrypt_directory_automation(ip, port, input_dir, encrypted_primary_key_path, encrypted_data_key_path, save_dir=None):
    print('[INFO] Encrypt Files Start...')
    if save_dir is None:
        save_dir = input_dir+'.encrypted'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    data_key = retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path)
    fernet = Fernet(data_key)
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        encrypted = fernet.encrypt(read_data_file(input_path))
        save_path = os.path.join(save_dir, file_name + '.encrypted')
        write_data_file(save_path, encrypted)
        print('[INFO] Encrypt Successfully! Encrypted Output Is ' + save_path)
    print('[INFO] Encrypted Files.')

def decrypt_csv_columns_automation(ip, port, encrypted_primary_key_path, encrypted_data_key_path, input_dir):
    from glob import glob
    import time
    print('[INFO] Column Decryption Start...')
    start = time.time()
    EXT = "*.csv"
    all_csv_files = [file
                     for p, subdir, files in os.walk(input_dir)
                     for file in glob(os.path.join(p, EXT))]
    data_key = retrieve_data_key_plaintext(ip, port,encrypted_primary_key_path, encrypted_data_key_path)
    fernet = Fernet(data_key)

    for csv_file in all_csv_files:
        data = csv.reader(open(csv_file,'r'))
        csvWriter = csv.writer(open(csv_file + '.col_decrypted', 'w', newline='\n'))
        csvWriter.writerow(next(data)) # Header
        for row in data:
            write_buffer = []
            for field in row:
                plaintext = fernet.decrypt(field.encode('ascii')).decode("utf-8")
                write_buffer.append(plaintext)
            csvWriter.writerow(write_buffer)
        print('[INFO] Decryption Finished. The Output Is ' + csv_file + '.col_decrypted')

    end = time.time()
    print('[INFO] Total Elapsed Time For Columns Decrytion: ' + str(end - start) + ' s')
