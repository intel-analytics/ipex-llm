# Import kms-client, csv And Decryption APIs To Get Datakey
import sys
from GetDataKeyPlaintext import decrypt_data_key as get_data_key_plaintext
import csv
from cryptography.fernet import Fernet
import os
from glob import glob
import argparse
import time

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
parser.add_argument('-path', '--path', type=str, help='path of the column encrypted CSV file', required=True)
parser.add_argument('-pkp', '--pkp', type=str, help='path of the primary key storage file', required=True)
parser.add_argument('-dkp', '--dkp', type=str, help='path of the data key storage file', required=True)

args = parser.parse_args()
ip = args.ip
port = args.port
path = args.path # CSV File Path
pkp = args.pkp
dkp = args.dkp

EXT = "*.csv"
all_csv_files = [file
                 for p, subdir, files in os.walk(path)
                 for file in glob(os.path.join(p, EXT))]

print('[INFO] Decryption Start...')

data_key = get_data_key_plaintext(ip, port, pkp, dkp) # Request Data Key For Decryption
fernet = Fernet(data_key) # Generate Decryption Method

for csv_file in all_csv_files:
    data = csv.reader(open(csv_file,'r'))
    csvWriter = csv.writer(open(csv_file + '.col_decrypted', 'w', newline='\n')) #Save Path
    csvWriter.writerow(next(data)) # Write CSV Header
    for row in data:
        write_buffer = []
        for field in row:
            plaintext = fernet.decrypt(field.encode('ascii')).decode("utf-8")
            write_buffer.append(plaintext)
        csvWriter.writerow(write_buffer)
    print('[INFO] One CSV File Decryption Finished. Current Output Is ' + csv_file + '.col_decrypted')

end = time.time()

print('[INFO] All Finished Successfully. Total Elapsed Time For Columns Decrytion: ' + str(end - start) + ' s')
