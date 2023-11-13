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

    data1 = pd.read_csv(csv_files[0], index_col=0)

    if len(csv_files)>1:
        data2 = pd.read_csv(csv_files[1], index_col=0)

        data1_df=pd.DataFrame(data1)
        data2_df=pd.DataFrame(data2)

        last1=[]
        diff1=[]
        last2=[]
        diff2=[]

        origin_column_1='1st token avg latency (ms)'
        origin_column_2='2+ avg latency (ms/token)'

        for data1_ind,data1_row in data1_df.iterrows():
            data1_model=data1_row['model'].strip()
            data1_input_output_pairs=data1_row['input/output tokens'].strip()
            flag=True

            for data2_ind,data2_row in data2_df.iterrows():
                data2_model=data2_row['model'].strip()
                data2_input_output_pairs=data2_row['input/output tokens'].strip()
                if data1_model==data2_model and data1_input_output_pairs==data2_input_output_pairs:
                    data1_temp=data1_row[origin_column_1]
                    data2_temp=data2_row[origin_column_1]
                    data3_temp=data1_row[origin_column_2]
                    data4_temp=data2_row[origin_column_2]

                    last1.append(data2_temp)
                    diff1.append(round((data2_temp-data1_temp)*100/data2_temp,2))
                    last2.append(data4_temp)
                    diff2.append(round((data4_temp-data3_temp)*100/data3_temp,2))

                    flag=False
                    break

            if flag:
                last1.append('')
                diff1.append('')
                last2.append('')
                diff2.append('')

        added_column_1='last1'
        added_column_2='diff1(%)'
        added_column_3='last2'
        added_column_4='diff2(%)'

        data1.insert(loc=3,column=added_column_1,value=last1)
        data1.insert(loc=4,column=added_column_2,value=diff1)
        data1.insert(loc=5,column=added_column_3,value=last2)
        data1.insert(loc=6,column=added_column_4,value=diff2)

    daily_html=csv_files[0].split(".")[0]+".html"
    data1.to_html(daily_html)

if __name__ == "__main__":
    sys.exit(main())