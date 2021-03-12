## AutoXgboost example
This example illustrates how to use autoxgboost to do classification and regression.

### Run steps
#### 1. Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

#### 2. Prepare dataset

For AutoXGBoostClassifier, download dataset from [here] (http://kt.ijs.si/elena_ikonomovska/data.html)
we will get file 'airline_14col.data.bz2', unzip it with

```bash
bzip2 -d airline_14col.data.bz2
```

we will get `airline_14col.data` for training

For AutoXGBoostRegressor, download dataset from [here] (https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/orca/automl/incd.csv)


#### 3. Run the AutoXGBoostClassifier example after pip install

data_path=... // training data path. Local file system is supported.

You can easily use the following commands to run this example:

```bash
python path/to/AutoXGBoostClassifier.py --path ${data_path}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.
