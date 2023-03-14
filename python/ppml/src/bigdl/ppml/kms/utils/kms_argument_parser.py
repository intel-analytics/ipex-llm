#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

class KmsArgumentParser:
    def __init__(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--app_id", type=str, default="simpleAPPID", help="app id")
        self.parser.add_argument("--api_key", type=str, default="simpleAPIKEY",help="app key")
        self.parser.add_argument("--kms_server_ip", type=str, help="ehsm or azure etc. kms server ip")
        self.parser.add_argument("--kms_server_port", type=str, help="ehsm or azure etc. kms server port")
        self.parser.add_argument("--kms_user_name", type=str, help="bigdl kms user name")
        self.parser.add_argument("--kms_user_token", type=str, help="bigdl kms user token")
        self.parser.add_argument("--vault", type=str, help="azure key vault name")
        self.parser.add_argument("--client_id", type=str, default="", help="azure client id")
        self.parser.add_argument("--primary_key_material", type=str, default="./primaryKeyPath", help="primary key path or name")
        self.parser.add_argument("--input_encrypt_mode", type=str, required=True, help="input encrypt mode")
        self.parser.add_argument("--output_encrypt_mode", type=str, required=True, help="output encrypt mode")
        self.parser.add_argument("--input_path", type=str, required=True, help="input path")
        self.parser.add_argument("--output_path", type=str, required=True, help="output path")
        self.parser.add_argument("--kms_type", type=str, default="SimpleKeyManagementService",
                help="SimpleKeyManagementService, EHSMKeyManagementService or AzureKeyManagementService")

    def get_arg_dict(self):
       args = self.parser.parse_args()
       arg_dict = vars(args)
       return arg_dict

