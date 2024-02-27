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
        elif val != 0.0 and not pd.isna(val) and is_last:
            return 'background-color: %s' % color3
    else:
        return ''

def nonzero_min(lst):
    non_zero_lst = [num for num in lst if num > 0.0]
    return min(non_zero_lst) if non_zero_lst else None

def is_diffs_within_normal_range(diff_en, diff_zh, threshold=5.0):
    return not any(diff < (-threshold) for diff in diff_en + diff_zh if isinstance(diff, float))

def add_to_dict(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

def best_in_dict(dict, key, value):
    if key in dict:
        best_value = nonzero_min(dict[key])
        if best_value < value or value <= 0.0:
            return best_value
        return value
    return value

def create_fp16_dict(fp16_path):
    fp16_df = pd.read_csv(fp16_path)
    fp16_dict = {}
    for _, row in fp16_df.iterrows():
        model = row['Model']
        # Formalize the data to have 2 decimal places
        fp16_dict[model] = {
            'en': "{:.2f}".format(row['en']),
            'zh': "{:.2f}".format(row['zh'])
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
    for task in ['en', 'zh']:
        latest_csv[f'{task}_FP16'] = latest_csv['Model'].apply(lambda model: fp16_dict.get(model, {}).get(task, 'N/A'))
        latest_csv[f'{task}_diff_FP16(%)'] = latest_csv.apply(lambda row: calculate_percentage_difference(row[task], row[f'{task}_FP16']), axis=1)

    if len(csv_files)>1:
        if args.baseline_path:
            previous_csv = pd.read_csv(args.baseline_path, index_col=0)
        else:
            previous_csv = pd.read_csv(csv_files[1], index_col=0)

        last_en=['']*len(latest_csv.index)
        diff_en=['']*len(latest_csv.index)
        last_zh=['']*len(latest_csv.index)
        diff_zh=['']*len(latest_csv.index)

        en='en'
        zh='zh'

        csv_dict = {}
        for csv_file in csv_files:
            current_csv = pd.read_csv(csv_file, index_col=0)
            for current_csv_ind,current_csv_row in current_csv.iterrows():
                current_csv_model=current_csv_row['Model'].strip()
                current_csv_precision=current_csv_row['Precision'].strip()
                current_csv_model_en=current_csv_model+'-'+current_csv_precision+'-'+'en'
                current_csv_model_zh=current_csv_model+'-'+current_csv_precision+'-'+'zh'
                add_to_dict(csv_dict, current_csv_model_en, current_csv_row[en])
                add_to_dict(csv_dict, current_csv_model_zh, current_csv_row[zh]) 

        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['Model'].strip()
            latest_csv_precision=latest_csv_row['Precision'].strip()
            latest_en=latest_csv_row[en]
            latest_zh=latest_csv_row[zh]

            in_previous_flag=False

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['Model'].strip()
                previous_csv_precision=previous_csv_row['Precision'].strip()

                if latest_csv_model==previous_csv_model and latest_csv_precision==previous_csv_precision:

                    previous_en=previous_csv_row[en]
                    previous_zh=previous_csv_row[zh]
                    if previous_en > 0.0 or previous_zh > 0.0:
                        last_en[latest_csv_ind]=previous_en
                        diff_en[latest_csv_ind]=round((latest_en-previous_en)*100/previous_en,2)
                        last_zh[latest_csv_ind]=previous_zh
                        diff_zh[latest_csv_ind]=round((latest_zh-previous_zh)*100/previous_zh,2)
                        in_previous_flag=True

            if not in_previous_flag:
                last_en[latest_csv_ind]=pd.NA
                diff_en[latest_csv_ind]=pd.NA
                last_zh[latest_csv_ind]=pd.NA
                diff_zh[latest_csv_ind]=pd.NA

        latest_csv.insert(loc=9,column='last_en',value=last_en)
        latest_csv.insert(loc=10,column='diff_en(%)',value=diff_en)
        latest_csv.insert(loc=11,column='last_zh',value=last_zh)
        latest_csv.insert(loc=12,column='diff_zh(%)',value=diff_zh)


        diffs_within_normal_range = is_diffs_within_normal_range(diff_en, diff_zh, threshold=highlight_threshold)

        subset1=['diff_en(%)','diff_zh(%)']
        
        columns={'en': '{:.2f}', 'zh': '{:.2f}', 'last_en': '{:.2f}', 'diff_en(%)': '{:.2f}',
                'last_zh': '{:.2f}', 'diff_zh(%)': '{:.2f}'}

        latest_csv.drop('Index', axis=1, inplace=True)

        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=3.0, is_last=True), subset=subset1)
        for task in ['en', 'zh']:
            styled_df = styled_df.applymap(lambda val: highlight_vals(val, max=highlight_threshold, is_last=False), subset=[f'{task}_diff_FP16(%)'])
        
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

    if args.baseline_path and not diffs_within_normal_range:
        print("The diffs are outside the normal range: %" + str(highlight_threshold))
        return 1 
    return 0

if __name__ == "__main__":
    sys.exit(main())
