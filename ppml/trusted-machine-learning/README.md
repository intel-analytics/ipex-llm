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
    - [3.2 LightGBM on Spark](#32-lightgbm-on-spark)


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
#### 1.2 Build Custom Image

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-bigdata#12-build-custom-image) to build a custom image with enclave signed by your private key.

### 2. Prepare SSL key and password

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#prepare-key-and-password) to prepare SSL key and password for secure container communication.

## Run machine learning example

The following shows how to run ML applications built on [SparkML](#2-run-spark-ml-application) or [LightGBM](#3-run-lightgbm-application) in a local or distributed (kubernetes/spark-on-kubernetes) fashion.

### 1. Configure K8S Environment

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#configure-the-environment) to create and configure K8S RBAC and secrets.

### 2. Run SparkML Application

#### 2.1 Start a client container

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-machine-learning#21-start-a-client-container) to start a cleint container.\
Configure the environment variables and the `DOCKER_IMAGE` should be the one built in [1.2 Build Machine Learning Custom Image](#12-build-machine-learning-custom-image)

#### 2.2 Submit local SparkML job 

MLlib toolkit in trusted-machine-learning porvides examples of some classic algorithms, like random forest, linear regression, gbt, K-Means etc. You can check the scripts in `/ppml/scripts` and execute one of them like this:

```bash 
bash scripts/classification/sgx/start-random-forest-classifier-on-local-sgx.sh
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
The SparkML toolkit also provides machine learning examples that use Kubernetes, and you can distinguish whether they use Kubernetes and what deploy mode they use by the script name. You can run one of the Kubernetes examples like this:

```bash
bash scripts/classification/native/start-random-forest-classifier-on-k8s-client-native.sh
```

### 3. Run LightGBM Application

#### 3.1 LightGBM Training on Kubernetes

The below illustrates how to run official example of [parallel learning](https://github.com/microsoft/LightGBM/tree/master/examples/parallel_learning#distributed-learning-example) as a reference, while user are allowed to run your custom applications (feature/data/voting parallel, or training/inference) by operating `LightGBM/trainer.conf` (then rebuild the image) and `LightGBM/kubernetes/upload-data.sh` (to specify your own data and data shards).

Go to the work directory:

```bash
cd LightGBM/kubernetes
```

Modify parameters in `start-lgbm-training.sh`:

```bash
export imageName=intelanalytics/bigdl-ppml-trusted-machine-learning-gramine-reference:2.3.0 # your custom image name if needed
export totalTrainerCount=2 # count of trainers as well as kubernetes pods
export trainerPort=12400 # base port number, while the real port can be adapted
export nfsMountPath=a_host_path_mounted_by_nfs_to_upload_data_before_training # the path you used to create kubernetes nfsvolumeclaim
export sgxEnabled=true # whether enable intel SGX
```

Then, start training by one command:

```bash
bash start-lgbm-training.sh
```

Then, you will find multiple models trained and saved by different trainers at `nfsMountPath` like the following:
```bash
ls $nfsMountPath/lgbm/lgbm-trainer-0/model.text
# you will see below:
model.text

ls $nfsMountPath/lgbm/lgbm-trainer-1/model.text
# you will see below:
model.text
```
The models are similar to each other because they have convergenced through Allreduce.

Finally, stop the trainer pods in consider of security:
```bash
bash uninstall-lgbm-trainer-from-k8s.sh
```

#### 3.2 LightGBM on Spark

we also provide an option to combine the two ML kits (SparkML and LightGBM), that seamlessly runs LighGBM applications on existing Spark cluster.

In such a scenario, LightGBM will utilize DataFrame etc. distribution abstractions to read and process big datasets in parallel, and ML pipeline etc. tools to do preprocessing and feature engineering efficiently. Meanwhile, Intel SGX, Gramine/Occlum LibOS, Key Management Service, and SSL/TLS etc. security tools are applied to protect key steps in cluster computing, such as parameter synchronization in training, model and data storage, and container runtime.

The Spark and LightGBM dependencies have already been installed in the custom image prepared in previous steps. For Occlum users, please use [trusted-big-data-ml occlum](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#trusted-big-data-ml-with-occlum).

##### End-to-End Fitting and Predication Examples of LightGBM on Spark

Here, We illustrate the progress with a Pyspark demo, and Scala is also supported.

For full-link protection, follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml#41-create-ppmlcontext) to deploy a KMS (Key Management Service) where you have many kinds of implementation type to choose, and generate a primary key firstly (the below uses `SimpleKeyManagementService`).

Next, before start training, download dataset [here](https://github.com/intel-analytics/BigDL/tree/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/lightGBM#uci-irisdata).

Moving on, there is an application to fit a LightGBM classification model, and save the trained model in cipher text, and then reload the encrypted model to predict. The source code can be seen [here](https://github.com/intel-analytics/BigDL/blob/main/python/ppml/example/lightgbm/encrypted_lightgbm_model_io.py), and you can follow the APIs to write your own privacy-preserving applications:

```python
sc = PPMLContext.init(...)
model = an instance of LightGBMClassficationModel/LightGBMRegressionModel/LightGBMRankerModel

# save trained model to file
sc.saveLightGBMModel(
    lightgbm_model = model,
    path = ...,
    crypto_mode = "PLAIN_TEXT"/"AES/CBC/PKCS5Padding"
)

# load model from file
classficationModel = sc.loadLightGBMClassificationModel(
    model_path = ...,
    crypto_mode = "PLAIN_TEXT"/"AES/CBC/PKCS5Padding")

regressionModel = sc.loadLightGBMRegressionModel(...)

rankerModel = sc.loadLightGBMRankerModel(...)
```

Now, it is time to **submit the spark job and start the LightGBM application**:

```bash
java \
-cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:${BIGDL_HOME}/jars/*" \
-Xmx512m \
org.apache.spark.deploy.SparkSubmit \
/ppml/examples/encrypted_lightgbm_model_io.py \
--app_id <your_simple_app_id> \
--api_key <your_simple_api_key> \
--primary_key_material <your_encrypted_primary_key_file_from_simple_kms> \
--input_path <path_to_iris.csv> \
--output_path <model_save_path> \
--output_encrypt_mode AES/CBC/PKCS5Padding \
--input_encrypt_mode PLAIN_TEXT
```

Parameter `--output_encrypt_mode` means how you want to save the trained model, and `--input_encrypt_mode` is the status of input dataset. Finally, you will get predications output from Spark driver, and find an encrypted classification model file saved on disk.

You can also submit a similar [Scala example](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/EncryptedLightGBMModelIO.scala) using **bigdl-ppml-submit.sh** mentioned before.
