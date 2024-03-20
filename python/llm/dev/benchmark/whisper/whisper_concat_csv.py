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

# Python program to concat CSVs

import os
import sys
import argparse
import pandas as pd
from datetime import date

def main():
    parser = argparse.ArgumentParser(description="concat .csv files")
    parser.add_argument("-i", "--input_path", type=str, dest="input_path",
                        help="The directory which stores the original CSV files", default="./")
    parser.add_argument("-o", "--output_path", type=str, dest="output_path",
                        help="The directory which stores the concated CSV file", default="./")
    
    args = parser.parse_args()

    csv_files = []
    for file_name in os.listdir(args.input_path):
        file_path = os.path.join(args.input_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            csv_files.append(file_path)
    csv_files.sort()

    merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    merged_df.reset_index(drop=True, inplace=True)

    today = date.today()
    csv_name = f'whisper-{today}.csv'
    output_file_path = os.path.join(args.output_path, csv_name)
    merged_df.to_csv(output_file_path)

if __name__ == "__main__":
    sys.exit(main())
