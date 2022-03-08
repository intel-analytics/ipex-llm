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
from email.policy import default
import numpy as np
from pyparsing import col
import click

"""
This script could split a dataset of csv to N parts with same size.
This script is for Vertical Federated Learning functionality test, we could split the data
into two parts and mock the Nclient (parties) holding data of different features.
"""
@click.command()
@click.argument('file_name')
@click.argument('num_pieces', type=int, default=2)
def vfl_split_dataset(file_name, num_pieces):
    with open(file_name, "r") as read_obj:
        f = open(file_name, "r")
        sample = f.readline().split(",")
        f.close()
        print(f"data has {len(sample)} columns")
        csv_reader = reader(read_obj)
        col_indices_list = []
        writer_list = []
        file_name = file_name.split('/')[-1].split(".")[-2]
        for i in range(num_pieces):
            col_indices_list.append([0] + [j for j in range(1 + i, len(sample), num_pieces)])
            writer_obj = open(f"{file_name}-{i}.csv", "w")
            writer_list.append(writer(writer_obj, delimiter=','))

        for i, row in enumerate(csv_reader):
            row = np.array(row)
            for j in range(num_pieces):
                writer_list[j].writerow(row[np.array(np.array(col_indices_list[j]))])


if __name__ == "__main__":
    vfl_split_dataset()
