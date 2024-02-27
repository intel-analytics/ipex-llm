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

def main():
    parser = argparse.ArgumentParser(description="convert .csv file to .html file")
    parser.add_argument("-f", "--folder_path", type=str, dest="folder_path",
                        help="The directory which stores the .csv file", default="/mnt/disk1/whisper_pr_gpu/")
    parser.add_argument("-t", "--threshold", type=float, dest="threshold",
                        help="the threshold of highlight values", default=1.0)
    args = parser.parse_args()

    csv_files = []
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            csv_files.append(file_path)
    csv_files.sort(reverse=True)

    latest_csv = pd.read_csv(csv_files[0], index_col=0)
    daily_html=csv_files[0].split(".")[0]+".html"

    if len(csv_files)>1:

        previous_csv = pd.read_csv(csv_files[1], index_col=0)

        last1=['']*len(latest_csv.index)
        diff1=['']*len(latest_csv.index)
        last2=['']*len(latest_csv.index)
        diff2=['']*len(latest_csv.index)

        WER='WER'
        RTF='RTF'

        for latest_csv_ind,latest_csv_row in latest_csv.iterrows():

            latest_csv_model=latest_csv_row['models'].strip()
            latest_csv_precision=latest_csv_row['precision'].strip()
            latest_WER=latest_csv_row[WER]
            latest_RTF=latest_csv_row[RTF]

            in_previous_flag=False

            for previous_csv_ind,previous_csv_row in previous_csv.iterrows():

                previous_csv_model=previous_csv_row['models'].strip()
                previous_csv_precision=previous_csv_row['precision'].strip()

                if latest_csv_model==previous_csv_model and latest_csv_precision==previous_csv_precision:

                    previous_WER=previous_csv_row[WER]
                    previous_RTF=previous_csv_row[RTF]
                    if previous_WER > 0.0 and previous_RTF > 0.0:
                        last1[latest_csv_ind]=previous_WER
                        diff1[latest_csv_ind]=round((previous_WER-latest_WER)*100/previous_WER,2)
                        last2[latest_csv_ind]=previous_RTF
                        diff2[latest_csv_ind]=round((previous_RTF-latest_RTF)*100/previous_RTF,2)
                        in_previous_flag=True

            if not in_previous_flag:
                last1[latest_csv_ind]=pd.NA
                diff1[latest_csv_ind]=pd.NA
                last2[latest_csv_ind]=pd.NA
                diff2[latest_csv_ind]=pd.NA

        latest_csv.insert(loc=4,column='last1',value=last1)
        latest_csv.insert(loc=5,column='diff1(%)',value=diff1)
        latest_csv.insert(loc=6,column='last2',value=last2)
        latest_csv.insert(loc=7,column='diff2(%)',value=diff2)

        subset1=['diff1(%)','diff2(%)']
        columns={'WER': '{:.6f}', 'RTF': '{:.6f}', 'last1': '{:.6f}', 'diff1(%)': '{:.6f}','last2': '{:.6f}', 'diff2(%)': '{:.6f}'}

        styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=1.0, color1='red', color2='green'), subset=subset1)
        html_output = styled_df.set_table_attributes("border=1").render()

        with open(daily_html, 'w') as f:
            f.write(html_output)
    else:
        latest_csv.to_html(daily_html)

    return 0

if __name__ == "__main__":
    sys.exit(main())