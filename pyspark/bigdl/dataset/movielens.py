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

import os

import zipfile
import numpy as np

from bigdl.dataset import base

SOURCE_URL = 'http://files.grouplens.org/datasets/movielens/'
def read_data_sets(data_dir):
    """
    Parse or download movielens 1m  data if train_dir is empty.

    :param data_dir: The directory storing the movielens data
    :return: a 2D numpy array with user index and item index in each row 
    """
    WHOLE_DATA = 'ml-1m.zip'
    local_file = base.maybe_download(WHOLE_DATA, data_dir, SOURCE_URL + WHOLE_DATA)
    zip_ref = zipfile.ZipFile(local_file, 'r')
    extracted_to = os.path.join(data_dir, "ml-1m")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (local_file, data_dir))
        zip_ref.extractall(data_dir)
        zip_ref.close()
    rating_files = os.path.join(extracted_to,"ratings.dat")

    rating_list = [i.strip().split("::") for i in open(rating_files,"r").readlines()]    
    movielens_data = np.array(rating_list).astype(int)
    return movielens_data 

def get_id_pairs(data_dir):
	movielens_data = read_data_sets(data_dir)
	return movielens_data[:, 0:2]

def get_id_ratings(data_dir):
	movielens_data = read_data_sets(data_dir)
	return movielens_data[:, 0:3]


if __name__ == "__main__":
    movielens_data = read_data_sets("/tmp/movielens/")
