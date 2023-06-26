# Secure LightGBM on Spark


we provide an option to combine the two ML kits (SparkML and LightGBM), that seamlessly runs LighGBM applications on existing Spark cluster.


In such a scenario, LightGBM will utilize DataFrame etc. distribution abstractions to read and process big datasets in parallel, and ML pipeline etc. tools to do preprocessing and feature engineering efficiently. Meanwhile, Intel SGX, Gramine/Occlum LibOS, Key Management Service, and SSL/TLS etc. security tools are applied to protect key steps in cluster computing, such as parameter synchronization in training, model and data storage, and container runtime.


The Spark and LightGBM dependencies have already been installed in the custom image prepared in previous steps. For Gramine user, please use [trusted-machine-learning image](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-machine-learning#gramine-machine-learning-toolkit), and [trusted-big-data-ml occlum](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#trusted-big-data-ml-with-occlum) for Occlum user.


## End-to-End LightGBM Fitting and Predication on Spark


Here, We illustrate the progress with a Pyspark demo, and Scala is also supported. 

### 1. Overall


- In the following example, a **PPMLContext** (entry for kinds of distributed APIs) is initialized first, and it will read CSV-ciphertext dataset with a schema specified in code, where encrypted data will be decrypted automatically and load into memory as DataFrame.


- Next, `transform` etc. APIs provided by **SparkML** kit are applied to do preprocessing like feature transformation and dataset splitting.


- Then, processed dataframe is feeded to **LightGBMClassifier**, and a training is invoked by `fit`.


- Finally, trained classification model is saved in ciphertext on disk, and we demonstrate that by loading the encrypted model into memory (and decrypted automatically) and using the reloaded model to predict on test set. The whole encryption/decryption process here applies the key specified by user configurations when submitting this Spark job.


For full-link protection, follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml#41-create-ppmlcontext) to deploy a KMS (Key Management Service) where you have many kinds of implementation type to choose, and generate a primary key firstly (the below uses `SimpleKeyManagementService`).


Next, before start training, download dataset [here](https://github.com/intel-analytics/BigDL/tree/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/lightGBM#uci-irisdata).


### 2. Start Pyspark Example


Moving on, there is an application to fit a LightGBM classification model, and save the trained model in ciphertext, and then reload the encrypted model to predict. The source code can be seen [here](https://github.com/intel-analytics/BigDL/blob/main/python/ppml/example/lightgbm/encrypted_lightgbm_model_io.py), and you can follow the APIs to write your own privacy-preserving applications:


```python
sc = PPMLContext.init(...)
model = an instance of LightGBMClassficationModel/LightGBMRegressionModel/LightGBMRankerModel

# save trained model to file
sc.saveLightGBMModel(
    lightgbm_model = model,
    path = ...,
    crypto_mode = "PLAIN_TEXT" / "AES/CBC/PKCS5Padding"
)

# load model from file
classficationModel = sc.loadLightGBMClassificationModel(
    model_path = ...,
    crypto_mode = "PLAIN_TEXT" / "AES/CBC/PKCS5Padding")

regressionModel = sc.loadLightGBMRegressionModel(...)

rankerModel = sc.loadLightGBMRankerModel(...)
```


**Mechanism:** BigDL PPML extract `Boosters` inside LightGBM models, serially convert them to `DataFrames` on Spark JVM, and encrypt them transparently through `codec in Hadoop IO compression`. The decryption is the deserialization process against it.


Now, it is time to **submit the spark job and start the LightGBM application**:


```bash
java \
-cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*" \
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

### 3. Start Scala Example

You can also submit a similar [Scala example](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/EncryptedLightGBMModelIO.scala), which has the same logic as the Pyspark one, using [PPML CLI](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli) like below:

```shell
bash bigdl-ppml-submit.sh \
 --master local[2] \
 --sgx-enabled false \
 --driver-memory 16g \
 --driver-cores 1 \
 --executor-memory 16g \
 --executor-cores 2 \
 --num-executors 8 \
 --conf spark.cores.max=8 \
 --conf spark.network.timeout=10000000 \
 --conf spark.executor.heartbeatInterval=10000000 \
 --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
 --conf spark.bigdl.primaryKey.defaultPK.plainText=<a_base64_256b_AES_key_string> \
 --class com.intel.analytics.bigdl.ppml.examples.EncryptedLightGBMModelIO \
 --jars ${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
 local://${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
<path_to_iris.csv>
```

For demo purpose, we directly apply a plaintext data key `spark.bigdl.primaryKey.defaultPK.plainText`, you can simply generate such a string by:

```shell
openssl enc -aes-256-cbc -k secret -P -md sha1
# you will get a key, and copy it to below field
echo <key_generated_above> | base64
```

Otherwise, only more safe key configurations are allowed in production environment, and please refer to [advanced Crypto in PPMLContext](https://github.com/intel-analytics/BigDL/tree/main/ppml#configurations-of-key-and-kms-in-ppmlcontext).
