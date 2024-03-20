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

import os
import pdb
import sys
import csv
import json
import datetime
import pandas as pd


if __name__ == '__main__':
    result_path = sys.argv[1]
    output_path = sys.argv[2]

    column_size = [25, 15, 10, 18, 15, 10, 10, 10]
    pad_string = lambda x, l: [i.ljust(j) for i, j in zip(x, l)]
    column_names = ["Model Name", "Precision", "STEM", "Social Science", "Humanities", "Other", "Hard", "Average"]

    print(f'\nDumping results for C-Eval score:\n')
    print(' '.join(pad_string(column_names, column_size)))
    print()

    file_lst = os.listdir(result_path)
    file_lst = [f'{result_path}/{i}' for i in file_lst]

    organized_dict = {}   # {'Qwen-7B': {'sym_int4': [], 'mixed_fp4': }}
    for file in file_lst:
        # Read the JSON file
        with open(file, 'r') as file:
            data = json.load(file)

            result_lst = [data['Model Name'], data['Precision']]

            result_lst += data['Results']

            # store in the organized dictionary
            try:
                organized_dict[data['Model Name']][data['Precision']] = result_lst
            except:
                organized_dict[data['Model Name']] = {}
                organized_dict[data['Model Name']][data['Precision']] = result_lst

    # define the print precision order
    model_order = ['chatglm2-6b', 'chinese-llama2-7b', 'Qwen-7B-Chat']
    precision_order = ['sym_int4', 'fp8_e5m2']
    
    # print the results
    for model_name in organized_dict.keys():
        for precision in precision_order:
            try:
                # print the result
                print(' '.join(pad_string(organized_dict[model_name][precision], column_size)))
            except KeyError:
                pass
        # separate between models
        print()
    
    # initialize the csv file
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f'results_{current_date}.csv'
    file_name = os.path.join(output_path, file_name) if output_path else file_name
    print('Writing to', file_name)

    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        headers = ["Model Name", "Precision", 'STEM', 'Social Science', 'Humanities', 'Other', 'Hard', 'Average']
        writer.writerow(headers)

        # print the results
        for model_name in model_order:
            for precision in precision_order:
                try:
                    # write the result to the csv row
                    writer.writerow(organized_dict[model_name][precision])
                except KeyError:
                    writer.writerow([model_name, precision]+[pd.NA for i in range(len(headers[2:]))])


