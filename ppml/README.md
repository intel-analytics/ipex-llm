## What is BigDL PPML
### States of data
In computing, data exists in three states: data in transit, data at rest, and data in use. 
1. Data being transferred on a network is “in transit”. tls
2. data in storage is “at rest”. transparent database encryption.
3. data being processed is “in use”. computing

### Confidential computing
[Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) is an emerging industry initiative focused on helping to secure data in use.

The efforts can enable encrypted data to be processed in memory while lowering the risk of exposing it to the rest of the system, thereby reducing the potential for sensitive data to be exposed while providing a higher degree of control and transparency for users. In multi-tenant cloud environments, where sensitive data is meant to be kept isolated from other privileged portions of the system stack, Intel® Software Guard Extensions (Intel® SGX) plays a large role in making this capability a reality.

### BigDL PPML
[BigDL](https://github.com/intel-analytics/BigDL) makes it easy for data scientists and data engineers to build end-to-end, distributed AI applications. [BigDL PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) provides a Trusted Cluster Environment for protecting the end-to-end Big Data AI pipeline. 

BigDL provides a distributed PPML platform for protecting the end-to-end Big Data AI pipeline (from data ingestion, data analysis, all the way to machine learning and deep learning). In particular, BigDL PPML provides a Trusted Cluster Environment, so as to run unmodified Big Data analysis and ML/DL programs in a secure fashion on (private or public) cloud:
- Compute and memory protected by SGX Enclaves
- Network communication protected by remote attestation and Transport Layer Security (TLS)
- Storage (e.g., data and model) protected by encryption
- Optional Federated Learning support


## What can BigDL PPML do
BigDL PPML allows users to run unmodified Big Data analysis and ML/DL programs (such as Apache Spark, Apache Flink, Tensorflow, PyTorch, etc.) in a secure fashion on (private or public) cloud.

### Data Analysis
### Machine Learning
With the trusted Big Data analytics and Machine Learning(ML)/Deep Learning(DL) support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, Spark MLlib, etc.) and distributed deep learning (using BigDL) in a secure and trusted fashion.

See the [PPML ML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html#trusted-big-data-analytics-and-ml) user guide for more details.

### Deep Learning
With the Trusted Realtime Compute and ML/DL support, users can run standard Flink stream processing and distributed DL model inference (using Cluster Serving in a secure and trusted fashion
See the [PPML DL](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html#trusted-realtime-compute-and-ml) user guide for more details.


## How to use BigDL PPML

## How to expand 
