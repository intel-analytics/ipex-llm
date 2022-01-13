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
path = all_csv_files[0]

data = csv.reader(open(path,'r'))

csvWriter = csv.writer(open(path + '.col_decrypted', 'w', newline='\n')) #Save Path

print('[INFO] Decryption Start...')
print('[INFO] If The First Line In Input CSV Is Not A Headr, The 18th Line Should Be Annotated')

# If The First Line In Input CSV Is The Header, Write The Header Directly. Otherwise, The Below Line Should Be Annotated
csvWriter.writerow(next(data))

data_key = get_data_key_plaintext(ip, port, pkp, dkp) # Request Data Key For Decryption

fernet = Fernet(data_key) # Generate Decryption Method

for row in data:
    write_buffer = []
    for i in range(len(row)):
        if i == 3:
            write_buffer.append(row[i])
        else:
            plaintext = fernet.decrypt(row[i].encode('ascii')).decode("utf-8")
            dpt = fernet.decrypt(plaintext.encode('ascii')).decode("utf-8")
        write_buffer.append(dpt)
    csvWriter.writerow(write_buffer)

end = time.time()

print('[INFO] Total Elapsed Time For Columns Decrytion: ' + str(end - start) + ' s')
print('[INFO] Decryption Finished. The Output Is ' + path + '.col_decrypted')
