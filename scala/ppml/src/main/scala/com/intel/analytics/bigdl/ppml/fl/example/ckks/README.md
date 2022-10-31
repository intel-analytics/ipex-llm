# VFL Logistic Regression with HE Example

This example show you how to create an end-to-end VFL Logistic Regression application with 2 clients and 1 server, using Homomorphic Encryption to protect data passing to FlServer on BigDL PPML.  
The targets and outputs of each client's linear will be protected by Homomorphic Encryption, clients will encrypt targets and outputs to cipher text, then send the cipher texts to FL server. FLServer will compute loss and grad using cipher texts, return the cipher-text results back to clients. After receive the cipher texts, Clients can use their private secrets to decrypt the returned cipher texts to loss and grad. The plain-text loss and grad will be given back linear to compute linear's grad, the rest work is just normal LR training.

### Data
We use [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) data in this example.

To simulate the scenario of two parties, we use select different features of Census data.

The original data has 15 columns. In preprocessing, some new feature are created from the combinations of some existed columns.

* data of client 1: `age`, `education`, `occupation`, cross columns: `edu_occ`, `age_edu_occ`
* data of client 2: `relationship`, `workclass`, `marital_status`

### Download BigDL assembly

Download BigDL assembly all in one jar from [BigDL-Release](https://bigdl.readthedocs.io/en/latest/doc/release.html), file name is bigdl-assembly-spark_[version]-jar-with-all-dependencies.jar

### Generate secret
`GenerateCkksSecret` will generate two secret file in given folder, one is `all_secret`, another is `compute_secret`.  
`all_secret`: contains all secret for both encryption and computing.
`compute_secret`: contains only computing secret for compute the cipher text, used on FLServer.
```bash
java -cp bigdl-assembly-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.ckks.GenerateCkksSecret [folder path]
```
This secret generation is not for production use, secret should be protected by Key Management Service.

### Start FLServer
Before starting server, modify the config file, `ppml-conf.yaml`, this application has 2 clients globally, and set the absolute path to ckks secret. So use following config:
```yaml
clientNum: 2
ckksSercetPath: /[absolute folder path]/compute_secret
```
Then start FLServer at server machine
```bash
java -cp bigdl-assembly-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.FLServer
```

## Start Local Trainers
Start the local Logistic Regression trainers at 2 training machines
```bash
java -cp bigdl-assembly-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.ckks.VflLogisticRegressionCkks
    -d [path to adult dataset]
    -i 1
    -s [path to all_secret]
# change -i 1 to -i 2 at client-2
```

The example will train the data and evaluate the training result.
