import os
import argparse
import collections
from typing import Tuple

import ray
from ray.data.aggregate import Mean, Std
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca import init_orca_context, stop_orca_context


class DataPreprocessor:
    """A Datasets-based preprocessor that fits scalers/encoders to the training
    dataset and transforms the training, testing, and inference datasets using
    those fitted scalers/encoders.
    """

    def __init__(self):
        # List of present fruits, used for one-hot encoding of fruit column.
        self.fruits = None
        # Mean and stddev stats used for standard scaling of the feature
        # columns.
        self.standard_stats = None

    def preprocess_train_data(self, ds: ray.data.Dataset
                              ) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
        print("\n\nPreprocessing training dataset.\n")
        return self._preprocess(ds, False)

    def preprocess_inference_data(self,
                                  df: ray.data.Dataset) -> ray.data.Dataset:
        print("\n\nPreprocessing inference dataset.\n")
        return self._preprocess(df, True)[0]

    def _preprocess(self, ds: ray.data.Dataset, inferencing: bool
                    ) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
        print(
            "\nStep 1: Dropping nulls, creating new_col, updating feature_1\n")

        def batch_transformer(df: pd.DataFrame):
            # Disable chained assignment warning.
            pd.options.mode.chained_assignment = None

            # Drop nulls.
            df = df.dropna(subset=["nullable_feature"])

            # Add new column.
            df["new_col"] = (
                df["feature_1"] - 2 * df["feature_2"] + df["feature_3"]) / 3.

            # Transform column.
            df["feature_1"] = 2. * df["feature_1"] + 0.1

            return df

        ds = ds.map_batches(batch_transformer, batch_format="pandas")

        print("\nStep 2: Precalculating fruit-grouped mean for new column and "
              "for one-hot encoding (latter only uses fruit groups)\n")
        agg_ds = ds.groupby("fruit").mean("feature_1")
        fruit_means = {
            r["fruit"]: r["mean(feature_1)"]
            for r in agg_ds.take_all()
        }

        print("\nStep 3: create mean_by_fruit as mean of feature_1 groupby "
              "fruit; one-hot encode fruit column\n")

        if inferencing:
            assert self.fruits is not None
        else:
            assert self.fruits is None
            self.fruits = list(fruit_means.keys())

        fruit_one_hots = {
            fruit: collections.defaultdict(int, **{fruit: 1})
            for fruit in self.fruits
        }

        def batch_transformer(df: pd.DataFrame):
            # Add column containing the feature_1-mean of the fruit groups.
            df["mean_by_fruit"] = df["fruit"].map(fruit_means)

            # One-hot encode the fruit column.
            for fruit, one_hot in fruit_one_hots.items():
                df[f"fruit_{fruit}"] = df["fruit"].map(one_hot)

            # Drop the fruit column, which is no longer needed.
            df.drop(columns="fruit", inplace=True)

            return df

        ds = ds.map_batches(batch_transformer, batch_format="pandas")

        if inferencing:
            print("\nStep 4: Standardize inference dataset\n")
            assert self.standard_stats is not None
        else:
            assert self.standard_stats is None

            print("\nStep 4a: Split training dataset into train-test split\n")

            # Split into train/test datasets.
            split_index = int(0.9 * ds.count())
            # Split into 90% training set, 10% test set.
            train_ds, test_ds = ds.split_at_indices([split_index])

            print("\nStep 4b: Precalculate training dataset stats for "
                  "standard scaling\n")
            # Calculate stats needed for standard scaling feature columns.
            feature_columns = [
                col for col in train_ds.schema().names if col != "label"
            ]
            standard_aggs = [
                agg(on=col) for col in feature_columns for agg in (Mean, Std)
            ]
            self.standard_stats = train_ds.aggregate(*standard_aggs)
            print("\nStep 4c: Standardize training dataset\n")

        # Standard scaling of feature columns.
        standard_stats = self.standard_stats

        def batch_standard_scaler(df: pd.DataFrame):
            def column_standard_scaler(s: pd.Series):
                if s.name == "label":
                    # Don't scale the label column.
                    return s
                s_mean = standard_stats[f"mean({s.name})"]
                s_std = standard_stats[f"std({s.name})"]
                return (s - s_mean) / s_std

            return df.transform(column_standard_scaler)

        if inferencing:
            # Apply standard scaling to inference dataset.
            inference_ds = ds.map_batches(
                batch_standard_scaler, batch_format="pandas")
            return inference_ds, None
        else:
            # Apply standard scaling to both training dataset and test dataset.
            train_ds = train_ds.map_batches(
                batch_standard_scaler, batch_format="pandas")
            test_ds = test_ds.map_batches(
                batch_standard_scaler, batch_format="pandas")
            return train_ds, test_ds


class Net(nn.Module):
    def __init__(self, n_layers, n_features, num_hidden, dropout_every,
                 drop_prob):
        super().__init__()
        self.n_layers = n_layers
        self.dropout_every = dropout_every
        self.drop_prob = drop_prob

        self.fc_input = nn.Linear(n_features, num_hidden)
        self.relu_input = nn.ReLU()

        for i in range(self.n_layers):
            layer = nn.Linear(num_hidden, num_hidden)
            relu = nn.ReLU()
            dropout = nn.Dropout(p=self.drop_prob)

            setattr(self, f"fc_{i}", layer)
            setattr(self, f"relu_{i}", relu)
            if i % self.dropout_every == 0:
                # only apply every few layers
                setattr(self, f"drop_{i}", dropout)
                self.add_module(f"drop_{i}", dropout)

            self.add_module(f"fc_{i}", layer)

        self.fc_output = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = self.fc_input(x)
        x = self.relu_input(x)

        for i in range(self.n_layers):
            x = getattr(self, f"fc_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            if i % self.dropout_every == 0:
                x = getattr(self, f"drop_{i}")(x)

        x = self.fc_output(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='PyTorch Ray Dataset Example')
    parser.add_argument('--runtime', type=str, default="ray",
                        help='The cluster mode, such as local, yarn, spark-submit or k8s.')
    parser.add_argument('--address', type=str, default="localhost:6379",
                        help='The address to use for Ray.')
    parser.add_argument('--backend', type=str, default="torch_distributed",
                        help='The backend of PyTorch Estimator; '
                            'bigdl, torch_distributed and spark are supported.')
    parser.add_argument('--batch_size', type=int, default=32, help='The training batch size')
    parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
    parser.add_argument('--data_dir', type=str, default="./data", help='The path of dataset')
    parser.add_argument('--smoke_test', type=str, default=False, help="Finish quickly for testing.")
    args = parser.parse_args()

    init_orca_context(runtime=args.runtime, address=args.address)

    def read_dataset(path) -> ray.data.Dataset:
        print(f"reading data from {path}")
        return ray.data.read_parquet(path, _spread_resource_prefix="node:") \
            .random_shuffle(_spread_resource_prefix="node:")

    if args.smoke_test:
        smoke_path = os.path.join(args.data_dir, "data_00000.parquet.snappy")
        smoke_dataset = read_dataset(smoke_path)
        train_dataset, test_dataset = DataPreprocessor().preprocess_train_data(smoke_dataset)
    else:
        dataset = read_dataset(args.data_dir)
        train_dataset, test_dataset = DataPreprocessor().preprocess_train_data(dataset)

    num_columns = len(train_dataset.schema().names)
    # remove label column and internal Arrow column.
    num_features = num_columns - 2
    
    config = {
        "num_hidden": 50,
        "num_layers": 3,
        "dropout_every": 5,
        "dropout_prob": 0.2,
        "num_features": num_features,
        "lr": 0.001
    }

    def model_creator(config):
        model = Net(n_layers=config["num_layers"],
                    n_features=config["num_features"],
                    num_hidden=config["num_hidden"],
                    dropout_every=config["dropout_every"],
                    drop_prob=config["dropout_prob"])
        model = model.double()
        return model

    def optim_creator(model, config):
        optimizer = optim.Adam(model.parameters(),
                               lr=config.get("lr", 0.001))
        return optimizer

    if args.backend == "torch_distributed":
        orca_estimator = Estimator.from_torch(model=model_creator,
                                              optimizer=optim_creator,
                                              loss=nn.BCEWithLogitsLoss(),
                                              metrics=[Accuracy()],
                                              model_dir=os.getcwd(),
                                              backend="torch_distributed",
                                              workers_per_node=2,
                                              use_tqdm=False,
                                              config=config)

        stats = orca_estimator.fit(train_dataset,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   label_cols="label")
        print(stats)
    else:
        raise ValueError("Only `torch.distributed` backend supports Ray Dataset Input!")

    stop_orca_context()


if __name__ == '__main__':
    main()