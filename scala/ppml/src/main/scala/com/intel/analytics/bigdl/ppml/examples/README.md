# Run Examples

#### 1.SimpleQuerySparkExample

This example do the following things.

1. Create a DataFrame(DF1) with three column(name, age, job) from a plain CSV file with PPMLContext.
2. Create a DataFrame(DF2) with three column(name, age, job) from a plain Parquet file with PPMLContext.
3. Do union on DF1 and DF2 and get a new DataFrame(DF3).
4. filter the people who's age between 20 to 40 on DF3 and get a DF4.
5. count how many people in each job on DF4 and get a DF5.
6. calculate the average age of people in each job on DF4 and get a DF6.
7. join DF5 and DF6 and get a result DataFrame.
8. Use PPMLContext, write the result DataFrame to a JSON file with encryption.

To run this example in spark local mode:

1. prepare the input data `people.csv` and `people.parquet` with following code.

   **Generate people.csv**

   ```python
   # generate_people_csv.py
   import sys
   import random
   jobs=['Developer', 'Engineer', 'Researcher']
   output_file = sys.argv[1]
   num_lines = int(sys.argv[2])
   with open(output_file, 'wb') as File:
       File.write("name,age,job\n".encode())
       cur_num_line = 0
       num_of_developer_age_between_20_and_40 = 0
       while(cur_num_line < num_lines):
           name_length = random.randint(3, 7)
           name = ''
           for i in range(name_length):
               name += chr(random.randint(97, 122))
           age=random.randint(18, 60)
           job=jobs[random.randint(0, 2)]
           if age <= 40 and age >= 20 and job == 'Developer':
               num_of_developer_age_between_20_and_40 += 1
           line = name + ',' + str(age) + ',' + job + "\n"
           File.write(line.encode())
           cur_num_line += 1
       print("Num of Developer age between 20,40 is " + str(num_of_developer_age_between_20_and_40))
   File.close()
   ```

   run the following command to generate `people.csv`

   ```bash
   python generate_people_csv.py </your/save/path/people.csv> <num_lines>
   ```

   

   **Generate people.parquet**

   ```python
   # generate_people_parquet.py
   import random
   import sys
   import os
   
   import pandas as pd
   import pyarrow as pa
   import pyarrow.parquet as pq
   
   
   def get_name():
       name_length = random.randint(3, 7)
       name = ''
       for i in range(name_length):
           name += chr(random.randint(97, 122))
   
       return name
   
   
   def generate_data(path, number):
       job = ['Developer', 'Engineer', 'Researcher']
   
       names = []
       ages = []
       jobs = []
       for i in range(number):
           names.append(get_name())
           ages.append(random.randint(18, 60))
           jobs.append(job[random.randint(0, 2)])
   
       df = pd.DataFrame({'name': names,
                          'age': ages,
                          'job': jobs})
   
       table = pa.Table.from_pandas(df)
       pq.write_table(table, os.path.join(path, "people.parquet"))
   
   
   if __name__ == '__main__':
       path = sys.argv[1]
       number = int(sys.argv[2])
       generate_data(path, number)
   ```

   run the following command to generate `people.parquet`

   ```bash
   python generate_people_parquet.py </your/save/path/people.parquet> <number>
   ```

2. select a KeyManagementService(`SimpleKeyManagementService` or `EHSMKeyManagementService`)

3. prepare a primaryKey and a dataKey(refer to [this](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/kms-utils/docker/README.md))

4. prepare a BigDL PPML Client Container(refer to PPML tutorial)

5. run the following command in the container

   make sure you /input/path contains both `people.csv` and `people.parquet`

- for `SimpleKeyManagementService` 

  ```bash
  /opt/jdk8/bin/java \
      -cp '/ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar:/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/*' -Xmx16g \
      org.apache.spark.deploy.SparkSubmit \
      --master local[4] \
      --executor-memory 8g \
      --driver-memory 8g \
      --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
      --conf spark.network.timeout=10000000 \
      --conf spark.executor.heartbeatInterval=10000000 \
      --verbose \
      --jars local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      --inputPath /your/input/path \
      --outputPath /your/output/path \
      --inputPartitionNum 8 \
      --outputPartitionNum 8 \
      --inputEncryptModeValue plain_text \
      --outputEncryptModeValue AES/CBC/PKCS5Padding \
      --primaryKeyPath /your/primary/key/primaryKey \
      --dataKeyPath /your/data/key/dataKey \
      --kmsType SimpleKeyManagementService \
      --simpleAPPID your_app_id \
      --simpleAPPKEY your_app_key
  ```

- for `EHSMKeyManagementService`

  ```bash
  /opt/jdk8/bin/java \
      -cp '/ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar:/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/*' -Xmx16g \
      org.apache.spark.deploy.SparkSubmit \
      --master local[4] \
      --executor-memory 8g \
      --driver-memory 8g \
      --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
      --conf spark.network.timeout=10000000 \
      --conf spark.executor.heartbeatInterval=10000000 \
      --verbose \
      --jars local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      --inputPath /your/input/path \
      --outputPath /your/output/path \
      --inputPartitionNum 8 \
      --outputPartitionNum 8 \
      --inputEncryptModeValue plain_text \
      --outputEncryptModeValue AES/CBC/PKCS5Padding \
      --primaryKeyPath /your/primary/key/primaryKey \
      --dataKeyPath /your/data/key/dataKey \
      --kmsType EHSMKeyManagementService \
      --kmsServerIP you_kms_server_ip \
      --kmsServerPort you_kms_server_port \
      --ehsmAPPID your_app_id \
      --ehsmAPPKEY your_app_key \
  ```

#### 2.xgbClassifierTrainingExampleOnCriteoClickLogsDataset

Run this example in spark local mode:

1.select a KeyManagementService(`SimpleKeyManagementService` or `EHSMKeyManagementService`)

2.prepare a primaryKey and a dataKey(refer to [this](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/kms-utils/docker/README.md))

3.prepare a BigDL PPML Client Container(refer to PPML tutorial)

4.run the following command in the container

> your input file can be CSV, JSON, PARQUET or other textfile with or without encryption. if input file is not encrypted, specify the `inputEncryptMode==plain_text`. else, for encrypted CSV, JSON and other textfile, specify the `inputEncryptMode==AES/CBC/PKCS5Padding`. for encrypted parquet file, specify the `inputEncryptMode==AES_GCM_CTR_V1 or AES_GCM_V1`.
>
> in this example, the input file is a plain CSV file

- for `SimpleKeyManagementService` 

  ```bash
  /opt/jdk8/bin/java \
      -cp '/ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar:/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/*' -Xmx16g \
      org.apache.spark.deploy.SparkSubmit \
      --master local[4] \
      --executor-memory 8g \
      --driver-memory 8g \
      --class com.intel.analytics.bigdl.ppml.examples.xgbClassifierTrainingExampleOnCriteoClickLogsDataset \
      --conf spark.network.timeout=10000000 \
      --conf spark.executor.heartbeatInterval=10000000 \
      --verbose \
      --jars local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      --trainingDataPath /your/training/data/path \
      --modelSavePath /your/model/save/path \
      --inputEncryptMode plain_text \
      --primaryKeyPath /your/primary/key/path/primaryKey \
      --dataKeyPath /your/data/key/path/dataKey \
      --kmsType SimpleKeyManagementService \
      --simpleAPPID your_app_id \
      --simpleAPPKEY your_app_key \
      --numThreads 1
  ```

- for `EHSMKeyManagementService`

  ```bash
  /opt/jdk8/bin/java \
      -cp '/ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar:/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/*' -Xmx16g \
      org.apache.spark.deploy.SparkSubmit \
      --master local[4] \
      --executor-memory 8g \
      --driver-memory 8g \
      --class com.intel.analytics.bigdl.ppml.examples.xgbClassifierTrainingExampleOnCriteoClickLogsDataset \
      --conf spark.network.timeout=10000000 \
      --conf spark.executor.heartbeatInterval=10000000 \
      --verbose \
      --jars local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      local:///ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar \
      --trainingDataPath /your/training/data/path \
      --modelSavePath /your/model/save/path \
      --inputEncryptMode plain_text \
      --primaryKeyPath /your/primary/key/path/primaryKey \
      --dataKeyPath /your/data/key/path/dataKey \
      --kmsType EHSMKeyManagementService \
      --kmsServerIP you_kms_server_ip \
      --kmsServerPort you_kms_server_port \
      --ehsmAPPID your_app_id \
      --ehsmAPPKEY your_app_key \
      --numThreads 1
  ```

  