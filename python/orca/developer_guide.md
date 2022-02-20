# BigDL Orca Developer Guide

This is a simple guide for developers to setup their development environment. You do **not** need to read this if you are just using Orca APIs.

## Prerequisite

- Linux OS
- JDK (e.g. `sudo apt-get install openjdk-8-jdk`)
- Maven (e.g. `sudo apt install maven`)
- Conda (Optional)


## Get BigDL source code

```bash
git clone https://github.com/intel-analytics/BigDL.git
```

## Build BigDL scala code

```bash
cd BigDL/scala
./make-dist.sh
```

After running the above code there will be a `BigDl/dist` directory consisting of the built artifact.

## Set up python environment

### PyTorch Development (pytorch_ray_estimator, pytorch_pyspark_estimator, not including pytorch_spark_estimator)

1. Install minimal dependencies

```bash
pip install pyspark==2.4.3 # ./make-dist.sh build against spark 2.4.3 by default
pip install ray==1.9.2
pip install aiohttp==3.7.4 # dependencis introduced by ray, latest version has api changes
pip install aioredis==1.3.1 # dependencis introduced by ray, latest version has api changes
pip install pytest pyarrow pandas
pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

2. Set a few environment variables

```bash
cd BigDL/python/orca
source dev/test/prepare_env.sh
```

3. Run tests to verify environment

```bash
pytest test/bigdl/orca/learn/ray/pytorch/test_estimator_pyspark_backend.py
```

4. Please raise an issue if the above steps does not work
