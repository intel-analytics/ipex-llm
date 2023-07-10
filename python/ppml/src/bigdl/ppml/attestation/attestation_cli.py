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

import attestation_service, quote_generator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", type=str, help='the url for attestation service', required=True)
    parser.add_argument("-t", "--as_type", type=str, help='the type of attestation service', default='BigDL')
    parser.add_argument("-i", "--app_id", type=str, help='the app id for attestation service', default='')
    parser.add_argument('-k', '--api_key', type=str, help='the api key for attestation service', default='')
    parser.add_argument('-O', '--quote_type', type=str, help='quote type', default='TDX')
    parser.add_argument('-o', '--policy_id', type=str, help='policy id', default='')
    parser.add_argument('-p', '--user_report', type=str, help='user report', default='ppml')
    parser.add_argument('--proxy', type=str, help='proxy', default='')
    args = parser.parse_args()

    proxies = {
        'http': args.proxy,
        'https': args.proxy
    }
    if args.quote_type == "TDX":
        quote = quote_generator.generate_tdx_quote(args.user_report)
    elif args.quote_type == "Gramine":
        quote = quote_generator.generate_gramine_quote(args.user_report)

    if args.as_type == "bigdl":
        attestation_result = attestation_service.bigdl_attestation_service(args.url, args.app_id, args.api_key, quote, args.policy_id)
    elif args.as_type == "amber":
        policy_list = args.policy_id.split(",")
        attestation_result = attestation_service.amber(args.url, args.api_key, quote, policy_list, proxies)

    if attestation_result == 0:
        print("Attestation Success!")
    if attestation_result == 1:
        print("WARNING: Attestation pass but BIOS or the software is out of date.")
        print("Attestation Success!")
    if attestation_result == -1:
        print("Attestation Failed!")
