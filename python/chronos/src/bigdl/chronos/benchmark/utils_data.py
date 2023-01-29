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


from sklearn.preprocessing import StandardScaler
from bigdl.chronos.data import get_public_dataset, gen_synthetic_data


def generate_data(args):
    """
    Generate dataset for training or inference.

    Args:
        args: is a ArgumentParser instance, inluding users input arguments.

    Returns:
        train_loader: is a dataset used to train.
        test_loader: is a dataset used to inference.
    """

    # read data
    if args.dataset == 'tsinghua_electricity':
        tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='tsinghua_electricity',
                                                                   with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)
    elif args.dataset == 'nyc_taxi':
        tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='nyc_taxi',
                                                                   with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)
    else:
        tsdata_train, tsdata_val, tsdata_test = gen_synthetic_data(with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)

    # preprocessing data
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute(mode="last")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))

    # transfer to tensorflow or torch dataset
    if args.framework == "tensorflow":
        for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
            tsdata.roll(lookback=args.lookback, horizon=args.horizon)
        train_loader = tsdata_train.to_tf_dataset(batch_size=args.training_batchsize)
        val_loader = tsdata_val.to_tf_dataset(batch_size=args.training_batchsize)
        test_loader = tsdata_test.to_tf_dataset(batch_size=args.inference_batchsize)
    else:
        add_args = {}
        if args.model == "autoformer":
            add_args = {'time_enc': True, 'label_len': int(args.lookback/2)}
        train_loader = tsdata_train.to_torch_data_loader(batch_size=args.training_batchsize,
                                                         roll=True,
                                                         lookback=args.lookback,
                                                         horizon=args.horizon,
                                                         **add_args)
        val_loader = tsdata_val.to_torch_data_loader(batch_size=args.training_batchsize,
                                                     roll=True,
                                                     lookback=args.lookback,
                                                     horizon=args.horizon,
                                                     **add_args)
        test_loader = tsdata_test.to_torch_data_loader(batch_size=args.inference_batchsize,
                                                       roll=True,
                                                       lookback=args.lookback,
                                                       horizon=args.horizon,
                                                       **add_args)

    return train_loader, val_loader, test_loader
