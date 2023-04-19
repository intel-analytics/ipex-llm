- [Gramine Machine Learning Toolkit](#gramine-machine-learning-toolkit)
  - [Before Running Code](#before-running-code)
    - [1. Build Docker Images](#1-build-docker-images)
      - [1.1 Build Machine Learning Base Image](#11-build-machine-learning-base-image)
      - [1.2 Build Machine Learning Custom Image](#12-build-machine-learning-custom-image)
    - [2. Prepare SSL key and password](#2-prepare-ssl-key-and-password)
- [Run machine learning example](#run-machine-learning-example)
  - [1. Configure K8S Environment](#1-configure-k8s-environment)
  - [2. Run Spark MLlib Application](#2-run-spark-mllib-application)
    - [2.1 Start a client container](#21-start-a-client-container)
    - [2.2 Submit local Spark Machine Learning job](#22-submit-local-spark-machine-learning-job)
    - [2.3 Submit Spark Machine Learning job with Kubernetes](#23-submit-spark-machine-learning-job-with-kubernetes)
  - [3. Run LightGBM Application](#3-run-lightgbm-application)
    - [3.1 LightGBM Training on Kubernetes](#31-lightgbm-training-on-kubernetes)


# Gramine Machine Learning Toolkit

This image contains Gramine and some popular Machine Learning frameworks including Spark and LightGBM. 

## Before Running Code
### 1. Build Docker Images
#### 1.1 Build Machine Learning Base Image

The machine learning base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following.

Before building your own base image, please modify the paths in `build-base-image.sh`. Then build the docker image with the following command.

```bash
./build-machine-learning-base-image.sh
```
#### 1.2 Build Machine Learning Custom Image

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-bigdata#12-build-custom-image) to build a custom image with enclave signed by your private key.

### 2. Prepare SSL key and password

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#prepare-key-and-password) to prepare SSL key and password for secure container communication.

## Run machine learning example

The following shows how to run ML applications with [Spark MLlib](#2-run-spark-mllib-application) or [LightGBM](#3-run-lightgbm-application) locally or on distributed kubernetes.

### 1. Configure K8S Environment

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#configure-the-environment) to create and configure K8S RBAC and secrets.

### 2. Run Spark MLlib Application

#### 2.1 Start a client container

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-machine-learning#21-start-a-client-container) to start a cleint container.\
Configure the environment variables and the `DOCKER_IMAGE` should be the one built in [1.2 Build Machine Learning Custom Image](#12-build-machine-learning-custom-image)

#### 2.2 Submit local Spark Machine Learning job 

MLlib toolkit in trusted-machine-learning porvides examples of some classic algorithms, like random forest, linear regression, gbt, K-Means etc. You can check the scripts in `/ppml/mllib/scripts` and execute one of them like this:

```bash 
bash mllib/scripts/classification/sgx/start-random-forest-classifier-on-local-sgx.sh
```

You can also submit a ML workload through [PPML CLI](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli):

```bash
bash bigdl-ppml-submit.sh \
     --name RandomForestClassifierExample \
     --sgx-enabled false \
     --master local[2] \
     --driver-memory 32g \
     --driver-cores 8 \
     --executor-memory 32g \
     --executor-cores 8 \
     --num-executors 2 \
     --class org.apache.spark.examples.ml.RandomForestClassifierExample \
     --verbose \
     --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
     local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```

Or run your own ML application with PPML CLI as below:
```bash 
export your_applicaiton_jar_path=...
export your_application_class_path=...
export your_application_arguments=...

./bigdl-ppml-submit.sh
    --master local[2] \
    --driver-memory 32g \
    --driver-cores 8 \
    --executor-memory 32g \
    --executor-cores 8 \
    --num-executors 2 \
    --class ${your_application_class_path} \
    --verbose \
    ${your_applicaiton_jar_path} \
    ${your_application_arguments}
```

#### 2.3 Submit Spark Machine Learning job with Kubernetes
The MLlib toolkit also provides machine learning examples that use Kubernetes, and you can distinguish whether they use Kubernetes and what deploy mode they use by the script name. You can run one of the Kubernetes examples like this:

```bash
bash mllib/scripts/classification/native/start-random-forest-classifier-on-k8s-client-native.sh
```

### 3. Run LightGBM Application

#### 3.1 LightGBM Training on Kubernetes

The below illustrates how to run official example of [parallel learning](https://github.com/microsoft/LightGBM/tree/master/examples/parallel_learning#distributed-learning-example) as a reference, while user are allowed to run your custom applications (feature/data/voting parallel, or training/inference) by operating `lgbm/train.conf` (then rebuild the image) and `lgbm/kubernetes/upload-data.sh` (to specify your own data and data shards).

Go to the work directory:

```bash
cd lgbm/kubernetes
```

Modify parameters in `install-lgbm-trainer.sh`:

```bash
export imageName=intelanalytics/bigdl-ppml-trusted-machine-learning-gramine-reference:2.3.0-SNAPSHOT # You custom image name if needed
export totalTrainerCount=2 # count of trainers as well as kubernetes pods
export trainerPort=12400 # base port number, while the real port can be adapted
export nfsMountPath=a_host_path_mounted_by_nfs_to_upload_data_before_training # the path you used to create kubernetes nfsvolumeclaim
```

Then, start training by one command:

```bash
bash install-lgbm-trainer.sh
```

Then, you will find multiple models trained and saved by different trainers at `nfsMountPath` like the following:
```bash
ls $nfsMountPath/lgbm/lgbm-trainer-0/model.text
model.text
ls $nfsMountPath/lgbm/lgbm-trainer-1/model.text
model.text
```
The models are similar to each other because they have convergenced through Allreduce.

Finally, stop the trainer pods in consider of security:
```bash
bash uninstall-lgbm-trainer.sh
```

