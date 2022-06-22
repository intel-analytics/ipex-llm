from bigdl.chronos.forecaster.tcn_forecaster import TCNForecaster
import pandas as pd
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler
from bigdl.orca.automl.metrics import Evaluator
from torch.utils.data.dataloader import DataLoader
import torch
import time
import numpy as np

if __name__ == '__main__':
    torch.set_num_threads(1)
    raw_df = pd.read_csv("data/electricity.csv")
    df = pd.DataFrame(pd.to_datetime(raw_df.date))
    for i in range(0, 320):
        df[str(i)] = raw_df[str(i)]
    df["OT"] = raw_df["OT"]

    target = []
    for i in range(0, 320):
        target.append(str(i))
    target.append("OT")

    look_back = 96
    horizon = 720

    tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="date", target_col=target, with_split=True, test_ratio=0.2, val_ratio=0.1)
    standard_scaler = StandardScaler()

    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.impute(mode="last")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))
            
    train_loader = tsdata_train.to_torch_data_loader(roll=True, lookback=look_back, horizon=horizon)
    test_loader = tsdata_test.to_torch_data_loader(roll=True, lookback=look_back, horizon=horizon)
    test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    forecaster = TCNForecaster(past_seq_len = look_back,
                            future_seq_len = horizon,
                            input_feature_num = 321,
                            output_feature_num = 321,
                            num_channels = [30] * 7,
                            repo_initialization = False,
                            kernel_size = 3, 
                            dropout = 0.1, 
                            lr = 0.001,
                            seed = 1)
    forecaster.fit(train_loader, epochs=1, batch_size=32)

    forecaster.num_processes = 1
    latency = []
    for x, y in test_loader:
        st = time.time()
        yhat = forecaster.predict(x.numpy())
        latency.append(time.time()-st)
    print("Inference latency is:", np.median(latency))

    forecaster.build_onnx(thread_num=1)
    onnx_latency = []
    for x, y in test_loader:
        st = time.time()
        y_pred = forecaster.predict_with_onnx(x.numpy())
        onnx_latency.append(time.time()-st)
    print("Inference latency with onnx is:", np.median(onnx_latency))