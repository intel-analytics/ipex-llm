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

import logging
import pandas as pd
import click

from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI

fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)


@click.command()
@click.option('--data_path', default="./data/diabetes-vfl-1.csv")
def run_client(data_path):
    init_fl_context(1)
    df_train = pd.read_csv(data_path)   
    
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    
    print(intersection[:5])

if __name__ == '__main__':
    run_client()
