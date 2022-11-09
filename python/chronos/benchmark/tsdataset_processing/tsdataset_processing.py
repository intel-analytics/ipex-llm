import time
import json
import argparse
from bigdl.chronos.data import get_public_dataset
from sklearn.preprocessing import StandardScaler

lookback = 48
horizon = 1

parser = argparse.ArgumentParser(description="TSDataset processing")
parser.add_argument("--name", default="nyc_taxi baseline", type=str)

def data_processing():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name="nyc_taxi")

    scaler = StandardScaler()

    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute()\
              .gen_dt_feature()\
              .scale(scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=lookback, horizon=horizon)


if __name__ == "__main__":
    args = parser.parse_args()

    process_start = time.time()
    data_processing()
    process_end = time.time()

    output = json.dumps({
        "config": args.name,
        "process_time": process_end - process_start
    })

    print(f'>>>{output}<<<')
