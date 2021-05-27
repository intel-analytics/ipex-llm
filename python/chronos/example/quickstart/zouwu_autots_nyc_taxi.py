#
# Copyright 2018 Analytics Zoo Authors.
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
import pandas as pd
from zoo.chronos.preprocessing.utils import train_val_test_split
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca import OrcaContext
from zoo.chronos.autots.forecast import AutoTSTrainer
from zoo.chronos.config.recipe import LSTMGridRandomRecipe
from zoo.chronos.autots.forecast import TSPipeline
import os
import argparse
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster.')
parser.add_argument("--num_workers", type=int, default=2,
                    help="The number of workers to be used in the cluster."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--cores", type=int, default=4,
                    help="The number of cpu cores you want to use on each node. "
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--memory", type=str, default="10g",
                    help="The memory you want to use on each node. "
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--data_dir", type=str, default="./nyc_taxi.csv",
                    help="the directory of electricity data file, you can download by running "
                         "wget https://raw.githubusercontent.com/numenta/NAB/v1.0/"
                         "data/realKnownCase/nyc_taxi.csv")

if __name__ == "__main__":

    args = parser.parse_args()

    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers

    init_orca_context(cluster_mode=args.cluster_mode,
                      cores=args.cores,
                      num_nodes=num_nodes,
                      memory=args.memory,
                      init_ray_on_spark=True
                      )

    # load the dataset. The downloaded dataframe contains two columns, "timestamp" and "value".
    df = pd.read_csv(args.data_dir, parse_dates=["timestamp"])

    # split the dataframe into train/validation/test set.
    train_df, val_df, test_df = train_val_test_split(df, val_ratio=0.1, test_ratio=0.1)

    trainer = AutoTSTrainer(dt_col="timestamp",  # the column name specifying datetime
                            target_col="value",  # the column name to predict
                            horizon=1,           # number of steps to look forward
                            extra_features_col=None  # list of column names as extra features
                            )

    ts_pipeline = trainer.fit(train_df, val_df,
                              recipe=LSTMGridRandomRecipe(
                                  num_rand_samples=1,
                                  epochs=1,
                                  look_back=6,
                                  batch_size=[64]),
                              metric="mse")

    # predict with the best trial
    pred_df = ts_pipeline.predict(test_df)

    # evaluate the result pipeline
    mse, smape = ts_pipeline.evaluate(test_df, metrics=["mse", "smape"])
    print("Evaluate: the mean square error is", mse)
    print("Evaluate: the smape value is", smape)

    # save & restore the pipeline
    with tempfile.TemporaryDirectory() as tempdirname:
        my_ppl_file_path = ts_pipeline.save(tempdirname + "saved_pipeline/nyc_taxi.ppl")
        loaded_ppl = TSPipeline.load(my_ppl_file_path)

    # Stop orca context when your program finishes
    stop_orca_context()
