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
import pandas as pd
from pathlib import Path

def highlight_vals(val, max=3.0, color1='red', color2='green', color3='yellow', is_last=False):
    if isinstance(val, float):
        if val > max:
            return 'background-color: %s' % color1
        elif val <= -max:
            return 'background-color: %s' % color2
        elif val != 0.0 and is_last:
            return 'background-color: %s' % color3
    else:
        return ''

def is_diffs_within_normal_range(diff_ppl_result, threshold=5.0):
    return not any(diff < (-threshold) for diff in diff_ppl_result if isinstance(diff, float))

def create_fp16_dict(fp16_path):
    fp16_df = pd.read_csv(fp16_path)
    fp16_dict = {}
    for _, row in fp16_df.iterrows():
        model = row['Model']
        # Formalize the data to have 2 decimal places
        fp16_dict[model] = {
            'ppl_result': "{:.2f}".format(row['ppl_result'])
        }
        
    return fp16_dict

def calculate_percentage_difference(current, fp16):
    if fp16 != 'N/A' and current != 'N/A' and float(fp16) != 0:
        return (float(current) - float(fp16)) / float(fp16) * 100
    else:
        return 'N/A'


def main():
    parser = argparse.ArgumentParser(description="convert .csv file to .html file")
    parser.add_argument("-f", "--folder_path", type=str, dest="folder_path",
                        help="The directory which stores the .csv file", default="/home/arda/yibo/BigDL/python/llm/dev/benchmark/harness")
    parser.add_argument("-t", "--threshold", type=float, dest="threshold",
                        help="the threshold of highlight values", default=3.0)
    parser.add_argument("-b", "--baseline_path", type=str, dest="baseline_path",
                        help="the baseline path which stores the baseline.csv file")
    args = parser.parse_args()

    # fp16.csv is downloaded previously under the parent folder of the folder_path
    parent_dir = Path(args.folder_path).parent
    fp16_path = os.path.join(parent_dir, 'fp16.csv')
    fp16_dict = create_fp16_dict(fp16_path)

    csv_files = []
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            csv_files.append(file_path)
    csv_files.sort(reverse=True)

    highlight_threshold=args.threshold
    
    latest_csv = pd.read_csv(csv_files[0], index_col=0)
    daily_html=csv_files[0].split(".")[0]+".html"

    # Reset index
    latest_csv.reset_index(inplace=True)

    diffs_within_normal_range = True

    # Add display of FP16 values for each model and add percentage difference column
    latest_csv['ppl_result_FP16'] = latest_csv['Model'].apply(lambda model: fp16_dict.get(model, {}).get('ppl_result', 'N/A'))
    latest_csv['ppl_result_diff_FP16(%)'] = latest_csv.apply(lambda row: calculate_percentage_difference(row['ppl_result'], row['ppl_result_FP16']), axis=1)

    if len(csv_files)>1:
        if args.baseline_path:
            previous_csv = pd.read_csv(args.baseline_path, index_col=0)
        else:
            previous_csv = pd.read_csv(csv_files[1], index_col=0)

        last_ppl_result=['']*len(latest_csv.index)
        diff_ppl_result=['']*len(latest_csv.index)

        ppl_result = 'ppl_result'
                
        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['Model'].strip()
            latest_csv_precision=latest_csv_row['Precision'].strip()
            latest_ppl_result=latest_csv_row[ppl_result]

            in_previous_flag=False

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['Model'].strip()
                previous_csv_precision=previous_csv_row['Precision'].strip()

                if latest_csv_model==previous_csv_model and latest_csv_precision==previous_csv_precision:

                    previous_ppl_result=previous_csv_row[ppl_result] 

                    if previous_ppl_result > 0.0:
                        last_ppl_result[latest_csv_ind]=previous_ppl_result
                        diff_ppl_result[latest_csv_ind]=round((latest_ppl_result-previous_ppl_result)*100/previous_ppl_result,2)
                        in_previous_flag=True

            if not in_previous_flag:
                last_ppl_result[latest_csv_ind]=pd.NA
                diff_ppl_result[latest_csv_ind]=pd.NA


        latest_csv.insert(loc=6,column='last_ppl_result',value=last_ppl_result)
        latest_csv.insert(loc=7,column='ppl_result_diff_last(%)',value=diff_ppl_result)


        diffs_within_normal_range = is_diffs_within_normal_range(diff_ppl_result, threshold=highlight_threshold)
        
        columns={'ppl_result': '{:.2f}', 'last_ppl_result': '{:.2f}', 'ppl_result_diff_last(%)': '{:.2f}'}
        latest_csv.drop('Index', axis=1, inplace=True)

        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=highlight_threshold, is_last=True), subset=['ppl_result_diff_last(%)'])
        styled_df = styled_df.applymap(lambda val: highlight_vals(val, max=highlight_threshold, is_last=False), subset=['ppl_result_diff_FP16(%)'])
 
    else:
        columns={'ppl_result': '{:.2f}'}
        latest_csv.drop('Index', axis=1, inplace=True)
        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=highlight_threshold, is_last=False), subset=['ppl_result_diff_FP16(%)'])
 
    # add css style to restrict width and wrap text
    styled_df.set_table_styles([{
        'selector': 'th, td',
        'props': [('max-width', '88px'), ('word-wrap', 'break-word')]
    }], overwrite=False)
   
    html_output = styled_df.set_table_attributes("border=1").to_html()
 
    with open(daily_html, 'w') as f:
        f.write(html_output)
 
    if args.baseline_path and not diffs_within_normal_range:
        print("The diffs are outside the normal range: %" + str(highlight_threshold))
        return 1
    return 0
 
if __name__ == "__main__":
    sys.exit(main())
