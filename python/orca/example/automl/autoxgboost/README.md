## AutoXGBoost example
This example illustrates how to use autoxgboost to do classification and regression.

### Run steps
#### 1. Install or download BigDL
Follow the instructions in [install guide](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-tuning.html#install) to install BigDL Orca.

Besides, you need to install xgboost with `pip install xgboost`.

#### 2. Prepare dataset

For AutoXGBoostClassifier, download dataset from [here](http://kt.ijs.si/elena_ikonomovska/data.html)
we will get file 'airline_14col.data.bz2', unzip it with

```bash
bzip2 -d airline_14col.data.bz2
```

we will get `airline_14col.data` for training

For AutoXGBoostRegressor, download dataset from [here](https://data.world/nrippner/cancer-analysis-hackathon-challenge)

You need to sign in and download `incd.csv` for training.

#### 3. Run the AutoXGBoostClassifier example after pip install

data_path = path/to/airline_14col.data. Local file system is supported.

You can easily use the following commands to run this example:

```bash
python path/to/AutoXGBoostClassifier.py --path ${data_path}
```

#### 4. Run the AutoXGBoostRegressor example after pip install

You can easily use the following commands to run this example:

```bash
python path/to/AutoXGBoostRegressor.py --path path/to/incd.csv
```

For running with parameter `-m sigopt`, you need to first obtain an SigOpt API token. You can register for SigOpt [here](https://app.sigopt.com/home) and the token could be found [here](https://app.sigopt.com/docs/overview/authentication) after logging in. To enable Ray to access the SigOpt API, you need to store the token as an environment variable as follows:

```bash
export SIGOPT_KEY=<YOUR-SIGOPT-API-TOKEN>
```

After running, you could view the comprehensive analysis of your experiment provided by SigOpt API [here](https://app.sigopt.com/home).
