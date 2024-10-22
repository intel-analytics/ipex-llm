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

# Python program to check if the number of lines in html meets expectation

import os
import sys
import argparse
import pandas as pd
from omegaconf import OmegaConf
from datetime import date

def main():
    parser = argparse.ArgumentParser(description="check if the number of lines in html meets expectation")
    parser.add_argument("-y", "--yaml-file", type=str, dest="yaml_name", help="name of yaml", required=True)
    parser.add_argument("--suffix", type=str, dest="file_suffix", help="the suffix of the csv_file")
    args = parser.parse_args()

    all_csv_files:list[str] = [file for file in os.listdir() if file.endswith('.csv')]

    conf = OmegaConf.load(args.yaml_name)
    test_apis : list[str] = conf['test_api']
    in_out_pairs : list[str] = conf['in_out_pairs']
    for api in test_apis:
        for in_out in in_out_pairs:
            csv_name_info:str = f"{in_out}-{api}-results-"
            csv_file = [file for file in all_csv_files if (csv_name_info in file) and file.endswith(f"_{args.file_suffix}.csv")][0]
    
            csv_dataframe = pd.read_csv(csv_file, index_col=0)
            actual_test_num = len(csv_dataframe)
            actual_test_cases = []
            for index, row in csv_dataframe.iterrows():
                actual_test_cases.append(row['model'] + ":" + row['input/output tokens'].split('-')[0] + ":" + str(row['batch_size']))

            
            all_test_cases = []
            for model in conf.repo_id:
                if not OmegaConf.is_list(conf["batch_size"]):
                    batch_list = [conf["batch_size"]]
                else:
                    batch_list = conf["batch_size"]
                for batch_size in batch_list:
                    model_id_input = model + ':' + in_out.split('-')[0] + ':' + str(batch_size)
                    all_test_cases.append(model_id_input)
            exclude_test_cases = []
            if 'exclude' in conf and conf['exclude'] is not None:
                exclude_test_cases = conf['exclude']
            expected_test_num = len(all_test_cases) - len(exclude_test_cases)
            if actual_test_num != expected_test_num:
                print("---------------The test cases should be tested!------------")
                for test_case in all_test_cases:
                    if test_case not in actual_test_cases and test_case not in exclude_test_cases:
                        print(test_case)
                raise ValueError("The above tests failed. Please check the errors in the log.")
            else:
                print("Results integrity check completed")

if __name__ == "__main__":
    sys.exit(main())
