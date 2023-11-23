# PPMLContext For PySpark

This is a tutorial about how to use `PPMLContext` in python to read/write files in multiple formats(csv, parquet, json etc.). `PPMLContext` provide the ability to save DataFrame as encrypted files and read encrypted files as a plain DataFrame or RDD.

### 0.How to submit PPMLContext task
The step 0 is tested based on intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.5.0-SNAPSHOT,
which needs to be configured according to the specific environment if using gramine or other image.
You can refer to [this example](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#bigdl-simplequery-e2e-example) for more detail.
1. set python home
```
export PYTHONHOME=/opt/python-occlum
```
2. set BigDL python lib
```
export PYTHONPATH=/opt/bigdl-2.5.0-SNAPSHOT/python-lib/
```
3. update *.py config. For example, encrypt_csv_util.py
```
ppml_args = {"kms_type": "SimpleKeyManagementService",
             "app_id": "123456654321",
             "api_key": "123456654321",
             "primary_key_material": "/opt/occlum_spark/data/key/simple_encrypted_primary_key",
             }
```
4. submit your task with spark jars and bigdl jars.
```
/usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -Djdk.lang.Process.launchMechanism=vfork \
                -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:$BIGDL_HOME/jars/*" \
                -Xmx512m org.apache.spark.deploy.SparkSubmit \
                /encrypt_csv_util.py /opt/occlum_spark/data/people.csv /opt/occlum_spark/data/people-encrypt/
```

### 1.Create a PPMLContext

`PPMLContext` who wraps a `SparkSession` and provides read functions to read encrypted data files to plain-text RDD or DataFrame, also provides write functions to save DataFrame to encrypted data files. 

So before you read/write files, you need to create a PPMLConext first. 

#### 1.1 create with app_name

This is the simplest way to create a `PPMLContext`. When you don't need to read/write encrypted files, you can use this way to create a `PPMLContext`.

```python
from bigdl.ppml.ppml_context import *
   
sc = PPMLContext("MyApp")
```

If you want to read/write encrypted files, then you need to provide more information.

#### 1.2 create with app_name & ppml_args

`ppml_args` is a dict, you need to provide the following parameters

- `kms_type`: the `KeyManagementService` you use, it can be `SimpleKeyManagementService` or `EHSMKeyManagementService`, the default `kms_type` is `SimpleKeyManagementService` 

if the `kms_type` is `SimpleKeyManagementService`, then need

- `simple_app_id`: the appId your KMS generated
- `simple_api_key`: the apiKey  your KMS generated
- `primary_key_path`: the path of your primaryKey
- `data_key_path`:  the path of your dataKey

if the `kms_type` is `EHSMKeyManagementService`, then need

- `kms_server_ip`: the server ip of your KMS
- `kms_server_port`: the server port of your KMS
- `ehsm_app_id`: the appId your KMS generated
- `ehsm_api_key`:  the apiKey  your KMS generated
- `primary_key_path`: the path of your primaryKey
- `data_key_path`:  the path of your dataKey

if the `kms_type` is `AzureKeyManagementService`, then need

- `azure_vault`: your azure vault name
- `azure_client_id(not necessary)`: your azure client id, default is empty.
- `primary_key_path`: the path of your primaryKey
- `data_key_path`:  the path of your dataKey

> How to generate appId, apiKey, primaryKey and dataKey, please refer to [this](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/kms-utils/docker/README.md)

Example

```python
# import
from bigdl.ppml.ppml_context import *

args = {"kms_type": "SimpleKeyManagementService",
        "simple_app_id": "123456",
        "simple_app_key": "123456",
        "primary_key_path": "/your/primary/key/path/primaryKey",
        "data_key_path": "/your/data/key/path/dataKey"
       }

sc = PPMLContext("MyApp", args)
```

#### 1.3 create with app_name & ppml_args & spark_conf

If you need to set Spark configurations, you can provide a `SparkConf` with Spark configurations to create a `PPMLContext`.

```python
from bigdl.ppml.ppml_context import *
from pyspark import SparkConf
   
ppml_args = {"kms_type": "SimpleKeyManagementService",
             "simple_app_id": "your_app_id",
             "simple_app_key": "your_app_key",
             "primary_key_path": "/your/primary/key/path/primaryKey",
             "data_key_path": "/your/data/key/path/dataKey"
            }
   
conf = SparkConf()
conf.setMaster("local[4]")

sc = PPMLContext("MyApp", ppml_args, conf)
```

### 2.Read & Write Files

you can read from a plain file or encrypted file, or write a DataFrame as plain or encrypted file, so you need to specify the `CryptoMode`:

- `plain_text`: no encryption
- `AES/CBC/PKCS5Padding`: for csv, json and text file
- `AES_GCM_V1`: for parquet only
- `AES_GCM_CTR_V1`: for parquet only

The following examples use `sc` to represent a initialized `PPMLContext`

#### 2.1 csv

Example

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain csv file and return a DataFrame
plain_csv_path = "/plain/csv/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).option("header", "true").csv(plain_csv_path)

# write a DataFrame as a plain csv file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.option("header", True)
.csv(plain_output_path)

# read a encrypted csv file and return a DataFrame
encrypted_csv_path = "/encrypted/csv/path"
df2 = sc.read(CryptoMode.AES_CBC_PKCS5PADDING).option("header", "true").csv(encrypted_csv_path)

# write a DataFrame as a encrypted csv file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_CBC_PKCS5PADDING)
.mode('overwrite')
.option("header", True)
.csv(encrypted_output_path)
```

the previous example use `CryptoMode.PLAIN_TEXT`/`CryptoMode.AES_CBC_PKCS5PADDING` as parameter, or you can just pass a string to `read/write` method.

Example

```python
df1 = sc.read("plain_text").option("header", "true").csv(plain_csv_path)
sc.write(df1, "plain_text")

df2 = sc.read("AES/CBC/PKCS5Padding").option("header", "true").csv(encrypted_csv_path)
sc.write(df2, "AES/CBC/PKCS5Padding")
```

you can use Enum Class `CryptoMode` or just a string interchangeably. 

**write mode**

there are 5 modes:

- `overwrite`: Overwrite existing data with the content of dataframe.
- `append`: Append content of the dataframe to existing data or table.
- `ignore`: Ignore current write operation if data / table already exists without any error.
- `error`: Throw an exception if data or table already exists.
- `errorifexists`: Throw an exception if data or table already exists.

#### 2.2 parquet

Example

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain parquet file and return a DataFrame
plain_parquet_path = "/plain/parquet/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).parquet(plain_parquet_path)

# write a DataFrame as a plain parquet file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.parquet(plain_output_path)

# read a encrypted parquet file and return a DataFrame
encrypted_parquet_path = "/encrypted/parquet/path"
df2 = sc.read(CryptoMode.AES_GCM_CTR_V1).parquet(encrypted_parquet_path)

# write a DataFrame as a encrypted parquet file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_GCM_CTR_V1)
.mode('overwrite')
.parquet(encrypted_output_path)
```

#### 2.3 json

Example

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain json file and return a DataFrame
plain_json_path = "/plain/json/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).json(plain_json_path)

# write a DataFrame as a plain json file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.json(plain_output_path)

# read a encrypted json file and return a DataFrame
encrypted_json_path = "/encrypted/parquet/path"
df2 = sc.read(CryptoMode.AES_CBC_PKCS5PADDING).json(encrypted_json_path)

# write a DataFrame as a encrypted parquet file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_CBC_PKCS5PADDING)
.mode('overwrite')
.json(encrypted_output_path)
```

#### 2.4 textfile

Example

```python
# import
from bigdl.ppml.ppml_context import *

# read from a plain csv file and return a RDD
plain_csv_path = "/plain/csv/path"
rdd1 = sc.textfile(plain_csv_path) # the default crypto_mode is "plain_text"

# read from a encrypted csv file and return a RDD
encrypted_csv_path = "/encrypted/csv/path"
rdd2 = sc.textfile(path=encrypted_csv_path, crypto_mode=CryptoMode.AES_CBC_PKCS5PADDING)
```



