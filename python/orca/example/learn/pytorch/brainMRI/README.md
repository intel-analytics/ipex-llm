## Trouble Shooting
- [x] no support of val set?
- [ ] Difficult to identify a customized loss function; for example, dice coefficient
- [ ] Unable to see the plot of matplotlib

## TODO
- [ ] Need to implement function to download dataset if it's not included in pytorch datasets
- [ ] bigdl backend development
- [ ] yarn development

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install torch
pip install torchvision
pip install matplotlib

# For bigdl backend:
pip install analytics-zoo  # 0.10.0.dev3 or above
pip install jep==3.9.0
pip install six cloudpickle

# For torch_distributed backend:
pip install analytics-zoo[ray]  # 0.10.0.dev3 or above

# For spark backend
pip install bigdl-orca
```

## Run on local after pip install

`bigdl` backend is still developing.

You can run with `torch_distributed` backend via:

```
python brainMRI.py --backend torch_distributed
```

You can run with `spark` backend via:

```
python brainMRI.py --backend spark
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```