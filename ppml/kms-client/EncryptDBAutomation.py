import sqlite3
import csv
import os
import argparse
import GeneratePrimaryKey,GenerateDataKey,EncryptFile
import shutil

def convert_db_to_csv(db_path):
    save_dir = db_path + '.csv'
    os.mkdir(save_dir)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = c.execute("select name from sqlite_master where type='table' order by name;")
    tables = [table[0] for table in query.fetchall()]
    for table_name in tables:
        columns = c.execute('PRAGMA table_info("' + table_name + '")')
        header = []
        for column in columns:
            header.append(column[1])
        csvWriter = csv.writer(open(os.path.join(save_dir,table_name + '.csv'), 'w', newline='\n'))
        csvWriter.writerow(header)
        c.execute('select * from ' + table_name + ';')
        rows = c.fetchall()
        csvWriter.writerows(rows)
        #for row in rows:
        #    csvWriter.writerow(row)

def encrypt_files(ip, port, pkp, dkp, input_dir, save_dir):
    for filename in os.listdir(input_dir):
        dfp = os.path.join(input_dir, filename)
        save_path = os.path.join(save_dir, filename + '.encrypted')
        EncryptFile.encrypt_file(ip, port, pkp, dkp, dfp, save_path)

parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
parser.add_argument('-dbp', '--dbp', type=str, help='path of the .db file to be processed', required=True)
args = parser.parse_args()
KEYWHIZ_SERVER_IP=args.ip
INPUT_DB_PATH = args.dbp

# Generate Keys And Use Them to Encyypt Files Outside SGX With KMS 
print("[INFO] Start To Process Target Data File Outside SGX...")
print("[INFO] Start To Convert The DB File To CSV Files...")
convert_db_to_csv(INPUT_DB_PATH)
print("[INFO] Start To Encrypt The Files...")
GeneratePrimaryKey.generate_primary_key(KEYWHIZ_SERVER_IP, '3000')
print('[INFO] Generate Data Key Start...')
GenerateDataKey.generate_data_key(KEYWHIZ_SERVER_IP, '3000', './encrypted_primary_key')
SAVE_DIR = INPUT_DB_PATH + '.encrypted'
os.mkdir(SAVE_DIR)
print('[INFO] Encrypt Files Start...')
encrypt_files(KEYWHIZ_SERVER_IP, '3000', './encrypted_primary_key', './encrypted_data_key', INPUT_DB_PATH + '.csv', SAVE_DIR)
print("[INFO] Encrypted Files Are Saved Under " + INPUT_DB_PATH + ".encrypted.")
shutil.rmtree(INPUT_DB_PATH + '.csv')
