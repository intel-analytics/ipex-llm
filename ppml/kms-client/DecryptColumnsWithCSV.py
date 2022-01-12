# Import kms-client, csv And Decryption APIs To Get Datakey
import sys
sys.path.append('/ppml/trusted-big-data-ml/work/kms-client')
from GetDataKeyPlaintext import decrypt_data_key as get_data_key_plaintext
import csv
from cryptography.fernet import Fernet
import os
from glob import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', type=str, help='path of the column encrypted CSV file', required=True)
args = parser.parse_args()

path = args.path # CSV File Path
EXT = "*.csv"
all_csv_files = [file
                 for p, subdir, files in os.walk(path)
                 for file in glob(os.path.join(p, EXT))]
path = all_csv_files[0]

data = csv.reader(open(path,'r'))

csvWriter = csv.writer(open(path + '.col_decrypted', 'w', newline='\n')) #Save Path

print('[INFO] Decryption Start...')
print('[INFO] If The First Line In Input CSV Is Not A Headr, The 18th Line Should Be Annotated')

# If The First Line In Input CSV Is The Header, Write The Header Directly. Otherwise, The Below Line Should Be Annotated
csvWriter.writerow(next(data))

data_key = get_data_key_plaintext("192.168.0.112","3000","/ppml/trusted-big-data-ml/encrypted_primary_key","/ppml/trusted-big-data-ml/encrypted_data_key") # Request Data Key For Decryption

fernet = Fernet(data_key) # Generate Decryption Method

for row in data:
    write_buffer = []
    for i in range(len(row)):
        if i == 3:
            write_buffer.append(row[i])
        else:
            #print(i)
            #print(row[i])
            plaintext = fernet.decrypt(row[i].encode('ascii')).decode("utf-8")
            #print(plaintext)
            dpt = fernet.decrypt(plaintext.encode('ascii')).decode("utf-8")
            #print(dpt)
            write_buffer.append(dpt)
    csvWriter.writerow(write_buffer)

print('[INFO] Decryption Finished. The Output Is ' + path + '.col_decrypted')
