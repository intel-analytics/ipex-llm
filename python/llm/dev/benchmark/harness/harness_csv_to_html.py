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

def highlight_vals(val, max=3.0, color1='red', color2='green'):
    if isinstance(val, float):
        if val > max:
            return 'background-color: %s' % color2
        elif val <= -max:
            return 'background-color: %s' % color1
    else:
        return ''

def nonzero_min(lst):
    non_zero_lst = [num for num in lst if num > 0.0]
    return min(non_zero_lst) if non_zero_lst else None

def is_diffs_within_normal_range(diff1, diff2, diff3, threshold=5.0):
    return not any(diff < (-threshold) for diff in diff1 + diff2 + diff3 if isinstance(diff, float))

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

def main():
    parser = argparse.ArgumentParser(description="convert .csv file to .html file")
    parser.add_argument("-f", "--folder_path", type=str, dest="folder_path",
                        help="The directory which stores the .csv file", default="/home/arda/yibo/BigDL/python/llm/dev/benchmark/harness")
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
    
    latest_csv = pd.read_csv(csv_files[0], index_col=0)
    daily_html=csv_files[0].split(".")[0]+".html"

    diffs_within_normal_range = True

    if len(csv_files)>1:
        if args.baseline_path:
            previous_csv = pd.read_csv(args.baseline_path, index_col=0)
        else:
            previous_csv = pd.read_csv(csv_files[1], index_col=0)

        last1=['']*len(latest_csv.index)
        diff1=['']*len(latest_csv.index)
        last2=['']*len(latest_csv.index)
        diff2=['']*len(latest_csv.index)
        last3=['']*len(latest_csv.index)
        diff3=['']*len(latest_csv.index)

        best_last1=['']*len(latest_csv.index)
        best_diff1=['']*len(latest_csv.index)
        best_last2=['']*len(latest_csv.index)
        best_diff2=['']*len(latest_csv.index)
        best_last3=['']*len(latest_csv.index)
        best_diff3=['']*len(latest_csv.index)

        Arc='Arc'
        TruthfulQA='TruthfulQA'
        Winogrande='Winogrande'

        csv_dict = {}
        for csv_file in csv_files:
            current_csv = pd.read_csv(csv_file, index_col=0)
            for current_csv_ind,current_csv_row in current_csv.iterrows():
                current_csv_model=current_csv_row['Model'].strip()
                current_csv_precision=current_csv_row['Precision'].strip()
                current_csv_model_arc=current_csv_model+'-'+current_csv_precision+'-'+'Arc'
                current_csv_model_truthfulqa=current_csv_model+'-'+current_csv_precision+'-'+'TruthfulQA'
                current_csv_model_winogrande=current_csv_model+'-'+current_csv_precision+'-'+'Winogrande'
                add_to_dict(csv_dict, current_csv_model_arc, current_csv_row[Arc])
                add_to_dict(csv_dict, current_csv_model_truthfulqa, current_csv_row[TruthfulQA])
                add_to_dict(csv_dict, current_csv_model_winogrande, current_csv_row[Winogrande])

        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['Model'].strip()
            latest_csv_precision=latest_csv_row['Precision'].strip()
            latest_arc=latest_csv_row[Arc]
            latest_truthfulqa=latest_csv_row[TruthfulQA]
            latest_winogrande=latest_csv_row[Winogrande]

            key1=latest_csv_model+'-'+latest_csv_precision+'-'+'Arc'
            key2=latest_csv_model+'-'+latest_csv_precision+'-'+'TruthfulQA'
            key3=latest_csv_model+'-'+latest_csv_precision+'-'+'Winogrande'

            best_last1_value=best_in_dict(csv_dict, key1, latest_arc)
            best_last2_value=best_in_dict(csv_dict, key2, latest_truthfulqa)
            best_last3_value=best_in_dict(csv_dict, key3, latest_winogrande)

            best_last1[latest_csv_ind]=best_last1_value
            best_diff1[latest_csv_ind]=round((best_last1_value-latest_arc)*100/best_last1_value,2)
            best_last2[latest_csv_ind]=best_last2_value
            best_diff2[latest_csv_ind]=round((best_last2_value-latest_truthfulqa)*100/best_last2_value,2)
            best_last3[latest_csv_ind]=best_last3_value
            best_diff3[latest_csv_ind]=round((best_last3_value-latest_winogrande)*100/best_last3_value,2)

            in_previous_flag=False

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['Model'].strip()
                previous_csv_precision=previous_csv_row['Precision'].strip()

                if latest_csv_model==previous_csv_model and latest_csv_precision==previous_csv_precision:

                    previous_arc=previous_csv_row[Arc]
                    previous_truthfulqa=previous_csv_row[TruthfulQA]
                    previous_winogrande=previous_csv_row[Winogrande]
                    if previous_arc > 0.0 and previous_truthfulqa > 0.0 and previous_winogrande > 0.0:
                        last1[latest_csv_ind]=previous_arc
                        diff1[latest_csv_ind]=round((previous_arc-latest_arc)*100/previous_arc,2)
                        last2[latest_csv_ind]=previous_truthfulqa
                        diff2[latest_csv_ind]=round((previous_truthfulqa-latest_truthfulqa)*100/previous_truthfulqa,2)
                        last3[latest_csv_ind]=previous_winogrande
                        diff3[latest_csv_ind]=round((previous_winogrande-latest_winogrande)*100/previous_winogrande,2)
                        in_previous_flag=True

            if not in_previous_flag:
                last1[latest_csv_ind]=pd.NA
                diff1[latest_csv_ind]=pd.NA
                last2[latest_csv_ind]=pd.NA
                diff2[latest_csv_ind]=pd.NA
                last3[latest_csv_ind]=pd.NA
                diff3[latest_csv_ind]=pd.NA

        latest_csv.insert(loc=5,column='last1',value=last1)
        latest_csv.insert(loc=6,column='diff1(%)',value=diff1)
        latest_csv.insert(loc=7,column='last2',value=last2)
        latest_csv.insert(loc=8,column='diff2(%)',value=diff2)
        latest_csv.insert(loc=9,column='last3',value=last3)
        latest_csv.insert(loc=10,column='diff3(%)',value=diff3)

        latest_csv.insert(loc=11,column='best 1',value=best_last1)
        latest_csv.insert(loc=12,column='best diff1(%)',value=best_diff1)
        latest_csv.insert(loc=13,column='best 2',value=best_last2)
        latest_csv.insert(loc=14,column='best diff2(%)',value=best_diff2)
        latest_csv.insert(loc=15,column='best 3',value=best_last3)
        latest_csv.insert(loc=16,column='best diff3(%)',value=best_diff3)

        diffs_within_normal_range = is_diffs_within_normal_range(diff1, diff2, diff3, threshold=highlight_threshold)

        subset1=['diff1(%)','diff2(%)','diff3(%)' ]
        subset2=['best diff1(%)','best diff2(%)','best diff3(%)']
        
        columns={'Arc': '{:.2f}', 'TruthfulQA': '{:.2f}', 'Winogrande': '{:.2f}', 'last1': '{:.2f}', 'diff1(%)': '{:.2f}',
                'last2': '{:.2f}', 'diff2(%)': '{:.2f}', 'last3': '{:.2f}', 'diff3(%)': '{:.2f}',
                'best 1': '{:.2f}', 'best diff1(%)': '{:.2f}', 'best 2': '{:.2f}', 'best diff2(%)': '{:.2f}', 'best 3': '{:.2f}', 'best diff3(%)': '{:.2f}'}

        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=3.0, color1='red', color2='green'), subset=subset1)
        styled_df = styled_df.applymap(lambda val: highlight_vals(val, max=3.0, color1='yellow'), subset=subset2)
        html_output = styled_df.set_table_attributes("border=1").render()

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
