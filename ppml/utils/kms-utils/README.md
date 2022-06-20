# PPML E2E Example Workflow
## Environment
* [Spark 3.x](https://spark.apache.org/downloads.html)
* [Maven](https://maven.apache.org/)

## Prepare

### Build Jar
  ```bash
  cd <path_to_repo_ppml-e2e-examples>/spark-encrypt-io
  mvn package
  ```
### Overall workflow of examples

Read below instructions before running examples:

- A CSV file named `people.csv` is inputed in some of the below examples, you can use [generate_people_csv.py](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.
- The input of SimpleQueryExample is `people.csv`, submit through java.
- The input of SimpleQuerySparkExample is `people.csv`, submit through spark.
- The input of SplitAndEncrypt is a path of a folder, which is filled with multiple CSVs.
- The input of SimpleEncryptIO is a single CSV. You can input any CSV files, and the example is to encrypt and save, and then decrypt and save as well.
- Detailed commands of each example are in the following.

## Run Examples
### [GenerateKeys](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/src/main/scala/com/intel/analytics/bigdl/ppml/examples/GenerateKeys.scala)

```bash
# SimpleKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
  --primaryKeyPath /the/path/you/want/to/put/encrypted/primary/key/at \
  --dataKeyPath /the/path/you/want/to/put/encrypted/data/key/at \
  --kmsType SimpleKeyManagementService
  --simpleAPPID $appid \
  --simpleAPPKEY $appkey
  
# EHSMKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
  --primaryKeyPath /the/path/you/want/to/put/encrypted/primary/key/at \
  --dataKeyPath /the/path/you/want/to/put/encrypted/data/key/at \
  --kmsType EHSMKeyManagementService
  --kmsServerIP your_ehsm_kms_server_ip \
  --kmsServerPort your_ehsm_kms_server_port \
  --ehsmAPPID your_ehsm_kms_appid \
  --ehsmAPPKEY your_ehsm_kms_appkey
```

### [LocalCryptoExample](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/src/main/scala/com/intel/analytics/bigdl/ppml/examples/LocalCryptoExample.scala)

```bash
# SimpleKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
    com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
    --inputPath $input_path \
    --primaryKeyPath /home/key/simple_encrypted_primary_key \
    --dataKeyPath /home/key/simple_encrypted_data_key \
    --kmsType SimpleKeyManagementService \
    --simpleAPPID $appid \
    --simpleAPPKEY $appkey
  
# EHSMKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
    com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
    --inputPath $input_path \
    --primaryKeyPath /home/key/ehsm_encrypted_primary_key \
    --dataKeyPath /home/key/ehsm_encrypted_data_key \
    --kmsType EHSMKeyManagementService \
    --kmsServerIP $EHSM_KMS_IP \
    --kmsServerPort $EHSM_KMS_PORT \
    --ehsmAPPID $appid \
    --ehsmAPPKEY $appkey
```
  
### [SplitAndEncrypt](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SplitAndEncrypt.scala)

```bash
# SimpleKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.examples.SplitAndEncrypt   \
  --inputPath /your/inputPath \
  --outputPath /your/outputPath \
  --outputPartitionNum 4 \
  --outputCryptoModeValue AES/CBC/PKCS5Padding \
  --primaryKeyPath /your/primaryKeyPath \
  --dataKeyPath /your/dataKeyPath \
  --kmsType SimpleKeyManagementService
  --simpleAPPID $appid \
  --simpleAPPKEY $appkey
  
# EHSMKeyManagementService
java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.examples.SplitAndEncrypt   \
  --inputPath /your/inputPath \
  --outputPath /your/outputPath \
  --outputPartitionNum 4 \
  --outputCryptoModeValue AES/CBC/PKCS5Padding \
  --primaryKeyPath /your/primaryKeyPath \
  --dataKeyPath /your/dataKeyPath \
  --kmsType SimpleKeyManagementService
  --kmsServerIP $EHSM_KMS_IP \
  --kmsServerPort $EHSM_KMS_PORT \
  --ehsmAPPID $appid \
  --ehsmAPPKEY $appkey
```

The tree of outputPath looks like this:
```bash
/outPutPath
├── file1
│   ├── split_0.csv
│   └── split_1.csv
└── file2
    ├── split_0.csv
    └── split_1.csv
```

###  [SimpleQueryExample](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQueryExample.scala)
```bash
# SimpleKeyManagementService on Spark cluster
$SPARK_HOME/bin/spark-submit \
  --master local[2] \
  --class com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
  --executor-memory 80g \
  --driver-memory 80g \
  ./target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
  --inputPath /yout/inputPath \
  --outputPath /yout/outputPath \
  --inputPartitionNum 8 \
  --outputPartitionNum 8 \
  --inputCryptoModeValue AES/CBC/PKCS5Padding \
  --outputCryptoModeValue plain_text \
  --primaryKeyPath /your/primaryKeyPath \
  --dataKeyPath /your/dataKeyPath \
  --kmsType SimpleKeyManagementService
  
  # SimpleKeyManagementService on host
  java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
    com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
    --inputPath $input_path \
    --outputPath $output_path \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputCryptoModeValue AES/CBC/PKCS5Padding \
    --outputCryptoModeValue plain_text \
    --primaryKeyPath /home/key/simple_encrypted_primary_key \
    --dataKeyPath /home/key/simple_encrypted_data_key \
    --kmsType SimpleKeyManagementService \
    --simpleAPPID $appid \
    --simpleAPPKEY $appkey
  
  # EHSMKeyManagementService on host
  java -cp target/spark-encrypt-io-0.3-SNAPSHOT-jar-with-dependencies.jar \
    com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
    --inputPath $input_path \
    --outputPath $output_path \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputCryptoModeValue AES/CBC/PKCS5Padding \
    --outputCryptoModeValue plain_text \
    --primaryKeyPath /home/key/ehsm_encrypted_primary_key \
    --dataKeyPath /home/key/ehsm_encrypted_data_key \
    --kmsType EHSMKeyManagementService \
    --kmsServerIP $EHSM_KMS_IP \
    --kmsServerPort $EHSM_KMS_PORT \
    --ehsmAPPID $appid \
```
