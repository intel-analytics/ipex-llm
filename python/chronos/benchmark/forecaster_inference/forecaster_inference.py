import time
import json
import argparse
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import get_public_dataset
from sklearn.preprocessing import StandardScaler

lookback, horizon = 48, 1

parser = argparse.ArgumentParser(description="TCNForecaster Training")
parser.add_argument("--name", default="TCNForecaster Training Baseline", type=str)
parser.add_argument("--accelerator", default="pytorch", type=str)

def create_data():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name="nyc_taxi")

    scaler = StandardScaler()

    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute()\
              .gen_dt_feature()\
              .scale(scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=lookback, horizon=horizon)

    return tsdata_train, tsdata_val, tsdata_test


def get_trained_forecaster(train_data, val_data):
    x, y = train_data.to_numpy()
    forecaster = TCNForecaster(past_seq_len=lookback,
                               future_seq_len=horizon,
                               input_feature_num=x.shape[-1],
                               output_feature_num=y.shape[-1],
                               seed=1)
    forecaster.fit((x,y), validation_data=val_data, epochs=3)
    return forecaster


def main():
    args = parser.parse_args()
    train_data, val_data, test_data = create_data()
    forecaster = get_trained_forecaster(train_data, val_data)

    accelerator_fn = {
        "pytorch": forecaster.predict,
        "onnx": forecaster.predict_with_onnx,
        "openvino": forecaster.predict_with_openvino
    }

    if args.accelerator == "onnx":
        forecaster.build_onnx()
    elif args.accelerator == "openvino":
        forecaster.build_openvino()

    predict_fn = accelerator_fn[args.accelerator]

    x_test, y_test = test_data.to_numpy()
    predict_start = time.time()
    for _ in range(100):
        y_hat = predict_fn(x_test)
    predict_end = time.time()

    output = json.dumps({
        "config": args.name,
        "inference_time": predict_end - predict_start
    })

    print(f'>>>{output}<<<')


if __name__ == "__main__":
    main()