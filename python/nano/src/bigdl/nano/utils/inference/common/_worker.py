import os
import pickle
import sys
import traceback
import cloudpickle

RETURN_FILENAME = 'return_value'


def main():
    param_file = sys.argv[1]
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    tmp_dir = os.path.dirname(param_file)
    return_value = params[0](*(params[1:]))
    with open(os.path.join(tmp_dir, RETURN_FILENAME), 'wb') as f:
        cloudpickle.dump(return_value, f)


if __name__ == "__main__":
    main()
