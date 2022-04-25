#
# Copyright 2021 The BigDL Authors.
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

from csv import reader, writer
import numpy as np
import click

"""
This script could split a dataset of csv to 2 parts. The odd columns, e.g. 1th, 3th, 5th, ...
would be split to one file and the even, e.g. 2th, 4th, 6th, ... would be split to another
This script is for Vertical Federated Learning functionality test, we could split the data
into two parts and mock the 2 client (parties) holding data of different features.
"""
@click.command()
@click.argument('file_name', '-f')
def vfl_split_dataset(file_name):
    with open(file_name, "r") as read_obj:
        f = open(file_name, "r")
        sample = f.readline().split(",")
        f.close()
        print(f"data has {len(sample)} columns")
        csv_reader = reader(read_obj)
        part1_col_idx = [0] + [i for i in range(1, len(sample), 2)]
        part2_col_idx = [0] + [i for i in range(2, len(sample), 2)]
        with open(f"{file_name}-1.csv", "w") as write_obj_1:
            with open(f"{file_name}-2.csv", "w") as write_obj_2:
                csv_writer_1 = writer(write_obj_1, delimiter=',')
                csv_writer_2 = writer(write_obj_2, delimiter=',')
                for i, row in enumerate(csv_reader):
                    row = np.array(row)
                    csv_writer_1.writerow(row[np.array(part1_col_idx)])
                    csv_writer_2.writerow(row[np.array(part2_col_idx)])


if __name__ == "__main__":
    vfl_split_dataset()
