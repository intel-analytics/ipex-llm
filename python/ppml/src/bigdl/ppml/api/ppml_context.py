from bigdl.ppml.api import *
import argparse


class PPMLContext(JavaValue):
    def __init__(self, jvalue=None, *args):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, *args)

    def load_keys(self, primary_key_path, data_key_path):
        callBigDlFunc(self.bigdl_type, "loadKeys", self.value, primary_key_path, data_key_path)

    def read(self, crypto_mode):
        df_reader = callBigDlFunc(self.bigdl_type, "read", self.value, crypto_mode)
        return EncryptedDataFrameReader(self.bigdl_type, df_reader)


class EncryptedDataFrameReader:
    def __init__(self, bigdl_type, df_reader):
        self.bigdl_type = bigdl_type
        self.df_reader = df_reader

    def option(self, key, value):
        self.df_reader = callBigDlFunc(self.bigdl_type, "option", self.df_reader, key, value)
        return self

    def csv(self, path):
        return callBigDlFunc(self.bigdl_type, "csv", self.df_reader, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_app_id", type=str, help="simple app id")
    parser.add_argument("--simple_app_key", type=str, help="simple app key")
    parser.add_argument("--primary_key_path", type=str, help="primary key path")
    parser.add_argument("--data_key_path", type=str, help="data key path")
    parser.add_argument("--input_encrypt_mode", type=str, help="input encrypt mode")
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--kms_type", type=str, default="SimpleKeyManagementService",
                        help="SimpleKeyManagementService or EHSMKeyManagementService")
    args = parser.parse_args()
    arg_dict = vars(args)

    ppml_context = PPMLContext(None, 'testApp', arg_dict)
    df = ppml_context.read(args.input_encrypt_mode) \
        .option("header", "true") \
        .csv(args.input_path)
    print(type(df))
