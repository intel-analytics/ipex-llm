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

def is_diffs_within_normal_range(diff1, diff2, threshold=5.0):
    return not any(diff < (-threshold) for diff in diff1 + diff2 if isinstance(diff, float))

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
                        help="The directory which stores the .csv file", default="/mnt/disk1/nightly_perf_gpu/")
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

        best_last1=['']*len(latest_csv.index)
        best_diff1=['']*len(latest_csv.index)
        best_last2=['']*len(latest_csv.index)
        best_diff2=['']*len(latest_csv.index)

        latency_1st_token='1st token avg latency (ms)'
        latency_2_avg='2+ avg latency (ms/token)'

        csv_dict = {}
        for csv_file in csv_files:
            current_csv = pd.read_csv(csv_file, index_col=0)
            for current_csv_ind,current_csv_row in current_csv.iterrows():
                current_csv_model=current_csv_row['model'].strip()
                current_csv_input_output_pairs=current_csv_row['input/output tokens'].strip()
                try: 
                    current_csv_batch_size=str(current_csv_row['batch_size'])
                    current_csv_model_input_1st=current_csv_model+'-'+current_csv_input_output_pairs+'-'+current_csv_batch_size+'-'+'1st'
                    current_csv_model_input_2nd=current_csv_model+'-'+current_csv_input_output_pairs+'-'+current_csv_batch_size+'-'+'2nd'
                    add_to_dict(csv_dict, current_csv_model_input_1st, current_csv_row[latency_1st_token])
                    add_to_dict(csv_dict, current_csv_model_input_2nd, current_csv_row[latency_2_avg])
                except KeyError:
                    #Old csv/html files didn't include 'batch_size' 
                    pass

        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['model'].strip()
            latest_csv_input_output_pairs=latest_csv_row['input/output tokens'].strip()
            latest_1st_token_latency=latest_csv_row[latency_1st_token]
            latest_2_avg_latency=latest_csv_row[latency_2_avg]
            latest_csv_batch_size=str(latest_csv_row['batch_size'])

            key1=latest_csv_model+'-'+latest_csv_input_output_pairs+'-'+latest_csv_batch_size+'-'+'1st'
            key2=latest_csv_model+'-'+latest_csv_input_output_pairs+'-'+latest_csv_batch_size+'-'+'2nd'

            best_last1_value=best_in_dict(csv_dict, key1, latest_1st_token_latency)
            best_last2_value=best_in_dict(csv_dict, key2, latest_2_avg_latency)

            best_last1[latest_csv_ind]=best_last1_value
            best_diff1[latest_csv_ind]=round((best_last1_value-latest_1st_token_latency)*100/best_last1_value,2)
            best_last2[latest_csv_ind]=best_last2_value
            best_diff2[latest_csv_ind]=round((best_last2_value-latest_2_avg_latency)*100/best_last2_value,2)

            in_previous_flag=False

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['model'].strip()
                previous_csv_input_output_pairs=previous_csv_row['input/output tokens'].strip()
                previous_csv_batch_size=str(previous_csv_row['batch_size'])

                if latest_csv_model==previous_csv_model and latest_csv_input_output_pairs==previous_csv_input_output_pairs and latest_csv_batch_size==previous_csv_batch_size:

                    previous_1st_token_latency=previous_csv_row[latency_1st_token]
                    previous_2_avg_latency=previous_csv_row[latency_2_avg]
                    if previous_1st_token_latency > 0.0 and previous_2_avg_latency > 0.0:
                        last1[latest_csv_ind]=previous_1st_token_latency
                        diff1[latest_csv_ind]=round((previous_1st_token_latency-latest_1st_token_latency)*100/previous_1st_token_latency,2)
                        last2[latest_csv_ind]=previous_2_avg_latency
                        diff2[latest_csv_ind]=round((previous_2_avg_latency-latest_2_avg_latency)*100/previous_2_avg_latency,2)
                        in_previous_flag=True

            if not in_previous_flag:
                last1[latest_csv_ind]=pd.NA
                diff1[latest_csv_ind]=pd.NA
                last2[latest_csv_ind]=pd.NA
                diff2[latest_csv_ind]=pd.NA

        latest_csv.insert(loc=3,column='last1',value=last1)
        latest_csv.insert(loc=4,column='diff1(%)',value=diff1)
        latest_csv.insert(loc=5,column='last2',value=last2)
        latest_csv.insert(loc=6,column='diff2(%)',value=diff2)

        latest_csv.insert(loc=7,column='best 1',value=best_last1)
        latest_csv.insert(loc=8,column='best diff1(%)',value=best_diff1)
        latest_csv.insert(loc=9,column='best 2',value=best_last2)
        latest_csv.insert(loc=10,column='best diff2(%)',value=best_diff2)

        diffs_within_normal_range = is_diffs_within_normal_range(diff1, diff2, threshold=highlight_threshold)

        subset1=['diff1(%)','diff2(%)']
        subset2=['best diff1(%)','best diff2(%)']
        columns={'1st token avg latency (ms)': '{:.2f}', '2+ avg latency (ms/token)': '{:.2f}', 'last1': '{:.2f}', 'diff1(%)': '{:.2f}',
                'last2': '{:.2f}', 'diff2(%)': '{:.2f}', 'encoder time (ms)': '{:.2f}', 'peak mem (GB)': '{:.2f}',
                'best 1': '{:.2f}', 'best diff1(%)': '{:.2f}', 'best 2': '{:.2f}', 'best diff2(%)': '{:.2f}', 'model loading time (s)': '{:.2f}'}

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
