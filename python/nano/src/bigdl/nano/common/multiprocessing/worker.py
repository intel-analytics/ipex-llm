import json
import os
import pickle
import sys


if __name__ == '__main__':
    temp_dir = sys.argv[1]
    with open(os.path.join(temp_dir, "train_ds_def.pkl"), 'rb') as f:
        train_ds_def = pickle.load(f)
    with open(os.path.join(temp_dir, "train_elem_spec.pkl"), 'rb') as f:
        train_elem_spec = pickle.load(f)
    with open(os.path.join(temp_dir, "val_ds_def.pkl"), 'rb') as f:
        val_ds_def = pickle.load(f)
    with open(os.path.join(temp_dir, "val_elem_spec.pkl"), 'rb') as f:
        val_elem_spec = pickle.load(f)
    with open(os.path.join(temp_dir, "fit_kwargs.pkl"), 'rb') as f:
        fit_kwargs = pickle.load(f)
    with open(os.path.join(temp_dir, "target.pkl"), 'rb') as f:
        target = pickle.load(f)

    history = target(temp_dir, train_ds_def, train_elem_spec,
               val_ds_def, val_elem_spec, fit_kwargs)
    tf_config = json.loads(os.environ["TF_CONFIG"])

    with open(os.path.join(temp_dir,
                           f"history_{tf_config['task']['index']}"),"wb") as f:
        pickle.dump(history, f)