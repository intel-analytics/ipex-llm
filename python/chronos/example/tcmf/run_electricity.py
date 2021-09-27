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
import argparse
from zoo.orca import init_orca_context, stop_orca_context
import numpy as np
from zoo.chronos.forecaster.tcmf_forecaster import TCMFForecaster
import tempfile
import logging
import sys
import os


def get_dummy_data():
    return np.random.randn(300, 480)


os.environ["KMP_AFFINITY"] = "disabled"

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
parser.add_argument("--data_dir", type=str,
                    help="the directory of electricity data file, you can download by running "
                         "https://github.com/rajatsen91/deepglo/blob/master/datasets/"
                         "download-data.sh. Note that we only need electricity.npy.")
parser.add_argument("--use_dummy_data", action='store_true', default=False,
                    help="Whether to use dummy data")
parser.add_argument("--smoke", action='store_true', default=False,
                    help="Whether to run smoke test")
parser.add_argument("--predict_local", action='store_true', default=False,
                    help="You can set this if want to run distributed training on yarn and "
                         "run distributed inference on local.")
parser.add_argument("--num_predict_cores", type=int, default=4,
                    help="The number of cores you want to use for prediction on local."
                         "You should only parse this arg if you set predict_local to true.")
parser.add_argument("--num_predict_workers", type=int, default=4,
                    help="The number of workers you want to use for prediction on local. "
                         "You should only parse this arg if you set predict_local to true.")

if __name__ == "__main__":

    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, num_nodes=num_nodes,
                      memory=args.memory, init_ray_on_spark=True)

    if not args.use_dummy_data:
        assert args.data_dir is not None, "--data_dir must be provided if not using dummy data"

    logger.info('Initalizing TCMFForecaster.')
    model = TCMFForecaster(
        vbsize=128,
        hbsize=256,
        num_channels_X=[32, 32, 32, 32, 32, 1],
        num_channels_Y=[32, 32, 32, 32, 32, 1],
        kernel_size=7,
        dropout=0.2,
        rank=64,
        kernel_size_Y=7,
        learning_rate=0.0005,
        normalize=False,
        use_time=True,
        svd=True,
    )
    ymat = np.load(args.data_dir) if not args.use_dummy_data else get_dummy_data()
    horizon = 24
    train_data = ymat[:, : -2 * horizon]
    target_data = ymat[:, -2 * horizon: -horizon]
    incr_target_data = ymat[:, -horizon:]

    logger.info('Start fitting.')
    model.fit({'y': train_data},
              val_len=24,
              start_date="2012-1-1",
              freq="H",
              covariates=None,
              dti=None,
              period=24,
              y_iters=1 if args.smoke else 10,
              init_FX_epoch=1 if args.smoke else 100,
              max_FX_epoch=1 if args.smoke else 300,
              max_TCN_epoch=1 if args.smoke else 300,
              alt_iters=2 if args.smoke else 10,
              num_workers=args.num_workers)
    logger.info('Fitting ends.')

    # you can save and load model as you want
    with tempfile.TemporaryDirectory() as tempdirname:
        model.save(tempdirname)
        loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)

    if args.predict_local:
        logger.info('Stopping context for yarn cluster and init context on local.')
        stop_orca_context()
        import ray
        ray.init(num_cpus=args.num_predict_cores)

    logger.info('Start prediction.')
    yhat = model.predict(horizon=horizon,
                         num_workers=args.num_predict_workers
                         if args.predict_local else args.num_workers)
    logger.info("Prediction ends")
    yhat = yhat["prediction"]
    target_value = dict({"y": target_data})

    # evaluate with prediction results
    from zoo.orca.automl.metrics import Evaluator
    evaluate_mse = Evaluator.evaluate("mse", target_data, yhat)

    # You can also evaluate directly without prediction results.
    mse, smape = model.evaluate(target_value=target_value, metric=['mse', 'smape'],
                                num_workers=args.num_predict_workers if args.predict_local
                                else args.num_workers)
    print(f"Evaluation results:\nmse: {mse}, \nsmape: {smape}")
    logger.info("Evaluation ends")

    # incremental fitting
    logger.info("Start fit incremental")
    model.fit_incremental({'y': target_data})
    logger.info("Start evaluation after fit incremental")
    incr_target_value = dict({"y": incr_target_data})
    mse, smape = model.evaluate(target_value=incr_target_value, metric=['mse', 'smape'],
                                num_workers=args.num_predict_workers
                                if args.predict_local else args.num_workers)
    print(f"Evaluation results after incremental fitting:\nmse: {mse}, \nsmape: {smape}")
    logger.info("Evaluation ends")

    stop_orca_context()
