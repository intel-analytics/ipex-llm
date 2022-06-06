from bigdl.ppml.fl import *
import argparse


class PPMLContext(JavaValue):
    def __init__(self, jvalue=None, *args):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, *args)

    def load_keys(self, primary_key_path, data_key_path):
        callBigDlFunc(self.bigdl_type, "loadKeys", self.value, primary_key_path, data_key_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_app_id", type=str, help="simple app id")
    parser.add_argument("--simple_app_key", type=str, help="simple app key")
    parser.add_argument("--primary_key_path", type=str, help="primary key path")
    parser.add_argument("--data_key_path", type=str, help="data key path")
    args = parser.parse_args()
    arg_dict = vars(args)

    ppml_context = PPMLContext(None, 'testApp', arg_dict)
    # ppml_context.load_keys(args.primary_key_path, args.data_key_path)
