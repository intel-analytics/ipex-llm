import json
import os
import pickle
import sys


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = pickle.load(f)

    with open(os.path.join(temp_dir, "target.pkl"), 'rb') as f:
        target = pickle.load(f)

    history = target(*args)
    tf_config = json.loads(os.environ["TF_CONFIG"])

    with open(os.path.join(temp_dir,
                           f"history_{tf_config['task']['index']}"),"wb") as f:
        pickle.dump(history, f)
