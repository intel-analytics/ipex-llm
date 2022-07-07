# PPMLContext For PySpark

### 1.Create a PPMLContext

#### 1.1 create with app_name

Example

```python
# import
from bigdl.ppml.ppml_context import *

sc = PPMLContext("MyApp")
```

#### 1.2 create with app_name & ppml_args

`ppml_args` is a dict, you need to provide the following parameters

- `kms_type`: the `KeyManagementService` you use, it can be `SimpleKeyManagementService` or `EHSMKeyManagementService`

if the `kms_type` is `SimpleKeyManagementService`, then need

- `simple_app_id`: the appId your KMS generated
- `simple_app_key`: the appKey  your KMS generated
- `primary_key_path`: the path of your primaryKey
- `data_key_path`:  the path of your dataKey

if the `kms_type` is `EHSMKeyManagementService`, then need

- `kms_server_ip`: the server ip of your KMS
- `kms_server_port`: the server port of your KMS
- `ehsm_app_id`: the appId your KMS generated
- `ehsm_app_key`:  the appKey  your KMS generated
- `primary_key_path`: the path of your primaryKey
- `data_key_path`:  the path of your dataKey

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

#### 1.3 create with app_name & ppml_args & SparkConf

Example

```python
# import
from bigdl.ppml.ppml_context import *
from pyspark import SparkConf

args = {"kms_type": "SimpleKeyManagementService",
        "simple_app_id": "123456",
        "simple_app_key": "123456",
        "primary_key_path": "/your/primary/key/path/primaryKey",
        "data_key_path": "/your/data/key/path/dataKey"
       }

# create a SparkConf
spark_conf = SparkConf()
spark_conf.setMaster("local[4]")

sc = PPMLContext("MyApp", args, spark_conf)
```

### 2.Read File

you can read from a plain file or encrypted file. so you need to specify the `CryptoMode`:

- `plain_text`: no encryption
- `AES/CBC/PKCS5Padding`: for csv, json and text file
- `AES_GCM_V1`: for parquet only
- `AES_GCM_CTR_V1`: for parquet only

#### 2.1 csv

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

# create a PPMLContext
sc = PPMLContext("MyApp", args)

# read a plain csv file and return a DataFrame
plain_csv_path = "/plain/csv/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).option("header", "true").csv(plain_csv_path)
# or
# df1 = sc.read("plain_text").option("header", "true").csv(plain_csv_path)

# read a encrypted csv file and return a DataFrame
encrypted_csv_path = "/encrypted/csv/path"
df2 = sc.read(CryptoMode.AES_CBC_PKCS5PADDING).option("header", "true").csv(encrypted_csv_path)
# or
# df2 = sc.read("AES_CBC_PKCS5PADDING").option("header", "true").csv(encrypted_csv_path)
```

#### 2.2 parquet

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

# create a PPMLContext
sc = PPMLContext("MyApp", args)

# read a plain parquet file and return a DataFrame
plain_parquet_path = "/plain/parquet/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).parquet(plain_parquet_path)
# or
# df1 = sc.read("plain_text").parquet(plain_parquet_path)

# read a encrypted parquet file and return a DataFrame
encrypted_parquet_path = "/encrypted/parquet/path"
df2 = sc.read(CryptoMode.AES_GCM_CTR_V1).parquet(encrypted_parquet_path)
# or
# df2 = sc.read("AES_GCM_CTR_V1").parquet(encrypted_parquet_path)
```

#### 2.3 textfile

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

# create a PPMLContext
sc = PPMLContext("MyApp", args)

# read from a plain csv file and return a RDD
plain_csv_path = "/plain/csv/path"
rdd1 = sc.textfile(plain_csv_path) # the default crypto_mode is "plain_text"

# read from a encrypted csv file and return a RDD
encrypted_csv_path = "/encrypted/csv/path"
rdd2 = sc.textfile(path=encrypted_csv_path, crypto_mode=CryptoMode.AES_CBC_PKCS5PADDING)
```

### 3.Write File

you can write as a plain file or encrypted file. so you also need to specify the `CryptoMode`

#### 3.1 csv

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

# create a PPMLContext
sc = PPMLContext("MyApp", args)

# DataFrame
df = ...

# write a DataFrame as a plain csv file
plain_output_path = "/plain/output/path"
sc.write(df, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.option("header", True)
.csv(plain_output_path)

# write a DataFrame as a encrypted csv file
encrypted_output_path = "/encrypted/output/path"
sc.write(df, CryptoMode.AES_CBC_PKCS5PADDING)
.mode('overwrite')
.option("header", True)
.csv(encrypted_output_path)
```

there are 5 modes:

- `overwrite`
- `append`
- `ignore`
- `error`
- `errorifexists`

#### 3.2 parquet

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

# create a PPMLContext
sc = PPMLContext("MyApp", args)

# DataFrame
df = ...

# write a DataFrame as a plain parquet file
plain_output_path = "/plain/output/path"
sc.write(df, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.parquet(plain_output_path)

# write a DataFrame as a encrypted parquet file
encrypted_output_path = "/encrypted/output/path"
sc.write(df, CryptoMode.AES_GCM_CTR_V1)
.mode('overwrite')
.parquet(encrypted_output_path)
```

