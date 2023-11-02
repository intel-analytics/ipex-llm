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

        origin_column_1='1st token avg latency (ms)'
        origin_column_2='2+ avg latency (ms/token)'

        added_column_1='last1'
        added_column_2='diff1(%)'
        added_column_3='last2'
        added_column_4='diff2(%)'

        data1.insert(loc=3,column=added_column_1,value=data2[origin_column_1])
        data1.insert(loc=4,column=added_column_2,value=round((data2[origin_column_1]-data1[origin_column_1])*100/data2[origin_column_1],2))
        data1.insert(loc=5,column=added_column_3,value=data2[origin_column_2])
        data1.insert(loc=6,column=added_column_4,value=round((data2[origin_column_2]-data1[origin_column_2])*100/data2[origin_column_2],2))

    daily_html=csv_files[0].split(".")[0]+".html"
    data1.to_html(daily_html)

if __name__ == "__main__":
    sys.exit(main())