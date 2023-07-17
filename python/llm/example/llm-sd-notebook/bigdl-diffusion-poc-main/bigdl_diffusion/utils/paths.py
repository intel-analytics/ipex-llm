# this module includes path utils for the Huggingface repository

import os

class NoPathException(Exception):
    pass

def get_local_path_from_repo_id(repo_id, models_root=os.getenv('HF_HOME')):
    # Applicable for diffusers models. Given a repo_id, get the local path of this model if exists
    
    if models_root is None:
        raise Exception("To use repo_id, you must set environmrnt variable `HF_HOME`.")

    repo_id, model_id = repo_id.split("/")
    # hardcode the diffusers path so that we only consider local models
    cache_dir = os.path.join(models_root, "diffusers", f"models--{repo_id}--{model_id}")
    model_path = get_snapshot_dir_from_cache_dir(cache_dir)
    return model_path

def get_snapshot_dir_from_cache_dir(cache_dir):
    # given a huggingface format cache dir, get the latest snapshot from it
    # TODO: probably add rolling strategy if any model fails
    assert os.path.exists(cache_dir), ">> Local model does not exist."
    snapshots_dir = os.path.join(cache_dir, "snapshots")
    snapshots = os.listdir(snapshots_dir)
    assert len(snapshots) != 0, f">> No models available, please download the model first"
    current_latest_snapshot = snapshots[0]
    current_latest_mtime = os.path.getmtime(os.path.join(snapshots_dir, current_latest_snapshot))
    for snap in snapshots:
        dir = os.path.join(snapshots_dir, snap)
        if os.path.getmtime(dir) > current_latest_mtime:
            current_latest_mtime = os.path.getmtime(dir)
            current_latest_snapshot = snap
    return os.path.join(snapshots_dir, current_latest_snapshot)