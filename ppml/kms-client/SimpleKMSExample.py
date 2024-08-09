import SimpleKeyManagementService, FileOperator
from cryptography.fernet import Fernet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dfp', '--dfp', type=str, help='path of the data file to encrypt', required=True)

data_file_path = parser.parse_args().dfp

# Prepare keys
SimpleKeyGenerator.enroll()
SimpleKeyGenerator.retrieve_primary_key('./encrypted_primary_key')
SimpleKeyGenerator.retrieve_data_key('./encrypted_primary_key','./encrypted_data_key')
data_key_plaintext = SimpleKeyGenerator.retrieve_data_key_plaintext('./encrypted_primary_key','./encrypted_data_key')

# Encrypt the data file
data_file_bytes_content = FileOperator.read_data_file(data_file_path)
fernet = Fernet(data_key_plaintext)
encrypted = fernet.encrypt(data_file_bytes_content)
encrypted_data_file_save_path = data_file_path + '.encrypted'
FileOperator.write_data_file(encrypted_data_file_save_path, encrypted)

# Decrypt the data file
encrypted_data_file_bytes_content =  FileOperator.read_data_file(encrypted_data_file_save_path)
decrypted = fernet.decrypt(encrypted_data_file_bytes_content)
decrypted_data_file_save_path = data_file_path + '.decrypted'
FileOperator.write_data_file(decrypted_data_file_save_path, decrypted)
