import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr
import argparse

import bigdl.orca
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.friesian.feature import FeatureTable

OrcaContext.log_output = True   # (this will display terminal's stdout and stderr in the Jupyter notebook).

cluster_mode = "local"
data_path = "/Users/yita/Documents/intel/data"

if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=1)
elif cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2)

dataset = {
    "ratings": ['userid', 'movieid',  'rating', 'timestamp'],
    "users": ["userid", "gender", "age", "occupation", "zip-code"],
    "movies": ["movieid", "title", "genres"]
}

tbl_dict = dict()
for data, cols in dataset.items():
    tbl = FeatureTable.read_csv(os.path.join(data_path, data + ".dat"),
                                delimiter=":", header=False)
    tmp_cols = tbl.columns[::2]
    tbl = tbl.select(tmp_cols)
    col_dict = {c[0]: c[1] for c in zip(tmp_cols, cols)}
    tbl = tbl.rename(col_dict)
    tbl_dict[data] = tbl

full_tbl = tbl_dict["ratings"].join(tbl_dict["movies"], "movieid").dropna()
# cast
full_tbl = full_tbl.cast(["rating"], "int")
train_tbl, test_tbl = full_tbl.random_split([0.85, 0.15], seed=1)






