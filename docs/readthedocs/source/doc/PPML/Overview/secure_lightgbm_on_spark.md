## Secure LightGBM on Spark


we provide an option to combine the two ML kits (SparkML and LightGBM), that seamlessly runs LighGBM applications on existing Spark cluster.


In such a scenario, LightGBM will utilize DataFrame etc. distribution abstractions to read and process big datasets in parallel, and ML pipeline etc. tools to do preprocessing and feature engineering efficiently. Meanwhile, Intel SGX, Gramine/Occlum LibOS, Key Management Service, and SSL/TLS etc. security tools are applied to protect key steps in cluster computing, such as parameter synchronization in training, model and data storage, and container runtime.


The Spark and LightGBM dependencies have already been installed in the custom image prepared in previous steps. For Occlum users, please use [trusted-big-data-ml occlum](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#trusted-big-data-ml-with-occlum).


### End-to-End Fitting and Predication Examples of LightGBM on Spark


Here, We illustrate the progress with a Pyspark demo, and Scala is also supported. 

#### Overall


In the following example, a **PPMLContext** (entry for kinds of distributed APIs) is initialized first, and it will read CSV-ciphertext dataset with a schema specified in code, where encrypted data will be decrypted automatically and load into memory as DataFrame.


Next, `transform` etc. APIs provided by **SparkML** kit are applied to do preprocessing like feature transformation and dataset splitting.


Then, processed dataframe is feeded to **LightGBMClassifier**, and a training is invoked by `fit`.


Finally, trained classification model is saved in ciphertext on disk, and we demonstrate that by loading the encrypted model into memory (and decrypted automatically) and using the reloaded model to predict on test set. The whole encryption/decryption process here applies the key specified by user configurations when submitting this Spark job.


### Start the Example


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


You can also submit a similar [Scala example](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/EncryptedLightGBMModelIO.scala) using **bigdl-ppml-submit.sh** mentioned before.
