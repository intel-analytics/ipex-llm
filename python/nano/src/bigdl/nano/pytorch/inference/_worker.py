import pickle
import sys
import traceback

import cloudpickle

if __name__ == "__main__":
    param_file = sys.argv[1]
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    try:
        print(cloudpickle.dumps(params[0](*(params[1:]))))
    except Exception as e:
        traceback.print_exc()
        print(f"----------worker exec failed ----------")
