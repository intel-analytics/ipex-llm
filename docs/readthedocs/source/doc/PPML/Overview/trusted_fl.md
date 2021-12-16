# Trusted FL (Federated Learning)

SGX-based End-to-end Trusted FL platform

## ID & Feature align

Before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage.

Let RID1 and RID2 be randomized ID from party 1 and party 2.

## Vertical FL

Vertical FL training across multi-parties with different features.

Key features:

* FL Server in SGX
    * ID & feature align
    * Forward & backward aggregation
* Training node in SGX

## Horizontal FL

Horizontal FL training across multi-parties.

Key features:

* FL Server in SGX
   * ID & feature align (optional)
   * Weight/Gradient Aggregation in SGX
* Training Worker in SGX
## Example 

### Prepare environment
#### SGX
TO ADD
#### Get jar ready
##### Build from source
```bash
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/scala
./make-dist.sh
```
the jar would be `BigDL/scala/ppml/target/bigdl-ppml...jar-with-dependencies.jar`
##### Download pre-build
```bash
wget
```
#### Config
If deploying PPML on cluster, need to overwrite config `./ppml-conf.yaml`. Default config (localhost:8980) would be used if no `ppml-conf.yaml` exists in the directory.
#### Start FL Server
```bash
java -cp com.intel.analytics.bigdl.ppml.FLServer
```
### HFL Logistic Regression
We provide an example demo in `BigDL/scala/ppml/demo`
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.HflLogisticRegression -d data/diabetes-hfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.HflLogisticRegression -d data/diabetes-hfl-2.csv
```
### VFL Logistic Regression
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d data/diabetes-vfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d data/diabetes-vfl-2.csv
```
## References

1. [Intel SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
2. Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. 2019. Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol. 10, 2, Article 12 (February 2019), 19 pages. DOI:https://doi.org/10.1145/3298981
