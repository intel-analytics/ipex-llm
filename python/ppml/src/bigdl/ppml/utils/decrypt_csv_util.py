from bigdl.ppml.ppml_context import *
from pyspark import SparkConf

ppml_args = {"kms_type": "SimpleKeyManagementService",
             "app_id": "123456654321",
             "api_key": "123456654321",
             "primary_key_material": "/opt/occlum_spark/data/key/simple_encrypted_primary_key",
             }

import sys
encrypt_csv_path = sys.argv[1]
plain_output_path = sys.argv[2]
conf = SparkConf()
conf.setMaster("local[4]")
sc = PPMLContext("MyApp", ppml_args, conf)
# import
from bigdl.ppml.ppml_context import *

# read an encrypted csv file and return a DataFrame
df1 = sc.read(CryptoMode.PLAIN_TEXT).option("header", "false").csv(encrypt_csv_path)
# write a DataFrame as a plain csv file
sc.write(df1, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').option("false", True).csv(plain_output_path)