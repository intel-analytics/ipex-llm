### Data
We use [Census]() data in this example

To simulate the scenario of two parties, we use select different features of Census data.

The original data has 15 columns. In preprocessing, some new feature are created from the combinations of some existed columns.

* data of client 1: `age`, `education`, `occupation`, cross columns: `edu_occ`, `age_edu_occ`
* data of client 2: `relationship`, `workclass`, `marital_status`

### Download BigDL assembly

Download BigDL assembly from [BigDL-Release](https://bigdl.readthedocs.io/en/latest/doc/release.html)

### Generate secret

```bash
java -cp bigdl-ppml-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.GenerateCkksSecret ckks.crt
```

### Start FLServer
Before starting server, modify the config file, `ppml-conf.yaml`, this application has 2 clients globally, and set the absolute path to ckks secret. So use following config:
```
worldSize: 2
ckksSercetPath: /[absolute path]/ckks.crt
```
Then start FLServer at server machine
```bash
java -cp bigdl-ppml-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.FLServer
```

## Start Local Trainers
Start the local Logistic Regression trainers at 2 training machines
```
java -cp bigdl-ppml-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.VflLogisticRegressionCkks
    -d [path to adult dataset]
    -i 1
    -s [path to ckks.crt]
# change -i 1 to -i 2 at client-2
```

The example will train the data and evaluate the training result.
