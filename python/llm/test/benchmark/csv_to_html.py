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

def highlight_vals(val, max=3.0):
    if val > max:
        return 'background-color: %s' % 'green'
    elif val < -max:
        return 'background-color: %s' % 'red'
    else:
        return ''

def main():
    parser = argparse.ArgumentParser(description="convert .csv file to .html file")
    parser.add_argument("-f", "--folder_path", type=str, dest="folder_path",
                        help="The directory which stores the .csv file", default="/mnt/disk1/nightly_perf/")
    args = parser.parse_args()

    csv_files = []
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            csv_files.append(file_path)
    csv_files.sort(reverse=True)

    latest_csv = pd.read_csv(csv_files[0], index_col=0)

    if len(csv_files)>1:
        previous_csv = pd.read_csv(csv_files[1], index_col=0)

        last1=['']*len(latest_csv.index)
        diff1=['']*len(latest_csv.index)
        last2=['']*len(latest_csv.index)
        diff2=['']*len(latest_csv.index)

        latency_1st_token='1st token avg latency (ms)'
        latency_2_avg='2+ avg latency (ms/token)'

        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['model'].strip()
            latest_csv_input_output_pairs=latest_csv_row['input/output tokens'].strip()
            latest_1st_token_latency=latest_csv_row[latency_1st_token]
            latest_2_avg_latency=latest_csv_row[latency_2_avg]

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['model'].strip()
                previous_csv_input_output_pairs=previous_csv_row['input/output tokens'].strip()

                if latest_csv_model==previous_csv_model and latest_csv_input_output_pairs==previous_csv_input_output_pairs:

                    previous_1st_token_latency=previous_csv_row[latency_1st_token]
                    previous_2_avg_latency=previous_csv_row[latency_2_avg]
                    last1[latest_csv_ind]=previous_1st_token_latency
                    diff1[latest_csv_ind]=round((previous_1st_token_latency-latest_1st_token_latency)*100/previous_1st_token_latency,2)
                    last2[latest_csv_ind]=previous_2_avg_latency
                    diff2[latest_csv_ind]=round((previous_2_avg_latency-latest_2_avg_latency)*100/previous_2_avg_latency,2)

        latest_csv.insert(loc=3,column='last1',value=last1)
        latest_csv.insert(loc=4,column='diff1(%)',value=diff1)
        latest_csv.insert(loc=5,column='last2',value=last2)
        latest_csv.insert(loc=6,column='diff2(%)',value=diff2)

    daily_html=csv_files[0].split(".")[0]+".html"

    subset=['diff1(%)','diff2(%)']
    columns={'1st token avg latency (ms)': '{:.2f}', '2+ avg latency (ms/token)': '{:.2f}', 'last1': '{:.2f}', 'diff1(%)': '{:.2f}',
             'last2': '{:.2f}', 'diff2(%)': '{:.2f}', 'encoder time (ms)': '{:.2f}'}

    with open(daily_html, 'w') as f:
        f.write(latest_csv.style.format(columns).applymap(highlight_vals, subset)
                        .set_table_attributes("border=1").render())

if __name__ == "__main__":
    sys.exit(main())