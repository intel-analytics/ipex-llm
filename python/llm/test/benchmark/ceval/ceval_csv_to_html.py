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

# Python program to convert CSV to HTML Table

import os
import sys
import argparse
import numpy as np
import pandas as pd

def highlight_vals(val, max=3.0, color1='red', color2='green', color3='yellow', is_last=False):
    if isinstance(val, float):
        if val > max:
            return 'background-color: %s' % color2
        elif val <= -max:
            return 'background-color: %s' % color1
        elif val != 0.0 and is_last:
            return 'background-color: %s' % color3
    else:
        return ''

def calculate_percentage_difference(cur_array, previous_array):
    new_array = []
    for i in range(len(cur_array)):
        if type(cur_array[i]) == type(pd.NA) or type(previous_array[i]) == type(pd.NA):
            new_array.append(pd.NA)
        else:
            new_array.append(round((cur_array[i]-previous_array[i])*100/previous_array[i], 2))
    return np.array(new_array)

def check_diffs_within_normal_range(latest_csv, highlight_set, threshold):
    within = True

    for column in highlight_set:
        for value in latest_csv[column]:
            if type(value) != type(pd.NA):
                within = within and abs(value) <= threshold
    
    return within


def main():
    parser = argparse.ArgumentParser(description="convert .csv file to .html file")
    parser.add_argument("-f", "--folder_path", type=str, dest="folder_path",
                        help="The directory which stores the .csv file", default="/home/arda/BigDL/python/llm/dev/benchmark/ceval")
    parser.add_argument("-t", "--threshold", type=float, dest="threshold",
                        help="the threshold of highlight values", default=3.0)
    parser.add_argument("-b", "--baseline_path", type=str, dest="baseline_path",
                        help="the baseline path which stores the baseline.csv file")
    args = parser.parse_args()

    csv_files = []
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            csv_files.append(file_path)
    csv_files.sort(reverse=True)

    highlight_threshold=args.threshold
    
    # get the newest csv file
    latest_csv = pd.read_csv(csv_files[0], index_col=0)

    # create daily html file
    daily_html=csv_files[0].split(".")[0]+".html"

    # add index column
    latest_csv.reset_index(inplace=True)

    # if found more than 1 csv file
    if len(csv_files)>1:
        if args.baseline_path:
            previous_csv = pd.read_csv(args.baseline_path, index_col=0)
        else:
            previous_csv = pd.read_csv(csv_files[1], index_col=0)

        subjects = ['STEM', 'Social Science', 'Humanities', 'Other', 'Hard', 'Average']
        precisions = ['sym_int4', 'fp8_e5m2']
        highlight_set = []

        insert_column = latest_csv.shape[-1]-1
        # in the make_csv step we will handle the missing values and make it pd.NA
        for subject in subjects:
            # insert last accuracy task
            latest_csv.insert(loc=insert_column, column=f'last_{subject}',
                              value=previous_csv[subject])

            # insert precentage difference between previous and current value
            latest_csv.insert(
                loc=insert_column+1,
                column=f'diff_{subject}(%)',
                value=calculate_percentage_difference(latest_csv[subject], previous_csv[subject]))
            # append in the highlight set
            highlight_set.append(f'diff_{subject}(%)')

            # update insert column
            insert_column += 2

        columns = {}
        for column in latest_csv.columns.values.tolist():
            columns[column] = '{:.2f}'

        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=3.0, is_last=True), subset=highlight_set)
        
        # add css style to restrict width and wrap text
        styled_df.set_table_styles([{
            'selector': 'th, td',
            'props': [('max-width', '88px'), ('word-wrap', 'break-word')]
        }], overwrite=False)
        
        html_output = styled_df.set_table_attributes("border=1").to_html()

        with open(daily_html, 'w') as f:
            f.write(html_output)
    else:
        latest_csv.to_html(daily_html)

    if args.baseline_path and not check_diffs_within_normal_range(latest_csv, highlight_set, highlight_threshold):
        print("The diffs are outside the normal range: %" + str(highlight_threshold))
        return 1 
    return 0

if __name__ == "__main__":
    sys.exit(main())
