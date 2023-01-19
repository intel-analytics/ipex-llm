import os
import pickle
import sys
import traceback
import cloudpickle

if __name__ == "__main__":
    param_file = sys.argv[1]
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    try:
        tmp_dir = os.path.dirname(param_file)
        return_value = params[0](*(params[1:]))
        with open(os.path.join(tmp_dir, 'return_value'), 'wb') as f:
            pickle.dump(return_value, f)
    except Exception as e:
        traceback.print_exc()
        print(f"----------worker exec failed ----------")
