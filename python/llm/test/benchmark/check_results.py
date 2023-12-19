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

def main():
    parser = argparse.ArgumentParser(description="check if the number of lines in html meets expectation")
    parser.add_argument("-c", "--csv_name", type=str, dest="csv_name",
                        help="name of csv", default="")
    parser.add_argument("-y", "--yaml_name", type=str, dest="yaml_name",
                        help="name of yaml", default="")
    args = parser.parse_args()

    csv_name = args.csv_name
    file_list = [file for file in os.listdir() if file.endswith('.csv') and csv_name in file]
    csv_dataframe = pd.read_csv(file_list[0], index_col=0)
    num_already_test_cases = len(csv_dataframe)
    already_test_cases = []
    for index, row in csv_dataframe.iterrows():
        already_test_cases.append(row['model'] + ":" + row['input/output tokens'].split('-')[0])

    yaml_name = args.yaml_name
    conf = OmegaConf.load(yaml_name)
    all_test_cases = []
    for model in conf.repo_id:
        for in_out in conf['in_out_pairs']:
            model_id_input = model + ':' + in_out.split('-')[0]
            all_test_cases.append(model_id_input)

    exclude_test_cases = []
    if 'exclude' in conf and conf['exclude'] is not None:
        exclude_test_cases = conf['exclude']

    num_expected_test_cases = len(all_test_cases) - len(exclude_test_cases)

    if num_already_test_cases != num_expected_test_cases:
        print("---------------The test cases should be tested!------------")
        for test_case in all_test_cases:
            if test_case not in already_test_cases and test_case not in exclude_test_cases:
                print(test_case)
        raise ValueError("The above test cases should be tested! Please check the saved csv carefully! ! !")

if __name__ == "__main__":
    sys.exit(main())