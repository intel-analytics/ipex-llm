#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.ppml.api import *
from bigdl.ppml.utils.log4Error import *
from enum import Enum

from pyspark.sql import SparkSession
from pyspark.ml.wrapper import JavaParams

def check(ppml_args, arg_name):
    try:
        value = ppml_args[arg_name]
        return value
    except KeyError:
        invalidInputError(False, "need argument " + arg_name)


class PPMLContext(JavaValue):
    def __init__(self, app_name, ppml_args=None, spark_conf=None):
        self.bigdl_type = "float"
        conf = {"spark.app.name": app_name,
                "spark.hadoop.io.compression.codecs": "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec"}
        if spark_conf:
            for (k, v) in spark_conf.getAll():
                conf[k] = v
        if ppml_args:
            kms_type = ppml_args.get("kms_type", "")
            conf["spark.bigdl.primaryKey.defaultKey.kms.type"] = kms_type
            if kms_type == "SimpleKeyManagementService":
                conf["spark.bigdl.primaryKey.defaultKey.kms.appId"] = check(ppml_args, "app_id")
                conf["spark.bigdl.primaryKey.defaultKey.kms.apiKey"] = check(ppml_args, "api_key")
                conf["spark.bigdl.primaryKey.defaultKey.material"] = check(ppml_args, "primary_key_material")
            elif kms_type == "EHSMKeyManagementService":
                conf["spark.bigdl.primaryKey.defaultKey.kms.ip"] = check(ppml_args, "kms_server_ip")
                conf["spark.bigdl.primaryKey.defaultKey.kms.port"] = check(ppml_args, "kms_server_port")
                conf["spark.bigdl.primaryKey.defaultKey.kms.id"] = check(ppml_args, "app_id")
                conf["spark.bigdl.primaryKey.defaultKey.kms.apiKey"] = check(ppml_args, "api_key")
                conf["spark.bigdl.primaryKey.defaultKey.material"] = check(ppml_args, "primary_key_material")
            elif kms_type == "AzureKeyManagementService":
                conf["spark.bigdl.primaryKey.defaultKey.kms.vault"] = check(ppml_args, "vault")
                conf["spark.bigdl.primaryKey.defaultKey.kms.clientId"] = ppml_args.get("client_id", "")
                conf["spark.bigdl.primaryKey.defaultKey.material"] = check(ppml_args, "primary_key_material")
            elif kms_type == "BigDLKeyManagementService":
                conf["spark.bigdl.primaryKey.defaultKey.kms.ip"] = check(ppml_args, "kms_server_ip")
                conf["spark.bigdl.primaryKey.defaultKey.kms.port"] = check(ppml_args, "kms_server_port")
                conf["spark.bigdl.primaryKey.defaultKey.kms.user"] = check(ppml_args, "kms_user_name")
                conf["spark.bigdl.primaryKey.defaultKey.kms.token"] = check(ppml_args, "kms_user_token")
                conf["spark.bigdl.primaryKey.defaultKey.material"] = check(ppml_args, "primary_key_material")
            elif kms_type == "":
                conf["spark.bigdl.primaryKey.defaultKey.plainText"] = check(ppml_args, "primary_key_plaintext")
            else:
                invalidInputError(False, "invalid KMS type.")

        spark_conf = init_spark_conf(conf)

        sc = SparkContext.getOrCreate(spark_conf)

        self.spark = SparkSession.builder.getOrCreate()
        args = [self.spark._jsparkSession]
        super().__init__(None, self.bigdl_type, *args)

    def read(self, crypto_mode, primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        df_reader = callBigDlFunc(self.bigdl_type, "read",
                                  self.value, crypto_mode, primary_key_name)
        return EncryptedDataFrameReader(self.bigdl_type, df_reader)

    def write(self, dataframe, crypto_mode, primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        df_writer = callBigDlFunc(self.bigdl_type, "write", self.value,
                                  dataframe, crypto_mode, primary_key_name)
        return EncryptedDataFrameWriter(self.bigdl_type, df_writer)

    def textfile(self, path, min_partitions=None,
                 crypto_mode = "plain_text", primary_key_name = ""):
        if min_partitions is None:
            min_partitions = self.spark.sparkContext.defaultMinPartitions
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        return callBigDlFunc(self.bigdl_type, "textFile",
                             self.value, path, min_partitions,
                             crypto_mode, primary_key_name)

    def sql(self, sqlText):
        return callBigDlFunc(self.bigdl_type, "sql",
                             self.value, sqlText)

    def saveLightGBMModel(self, lightgbm_model, path,
            crypto_mode = "plain_text", primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        return callBigDlFunc(self.bigdl_type, "saveLightGBMModel",
                self.value, lightgbm_model._java_obj,
                path, crypto_mode, primary_key_name)

    def loadLightGBMClassificationModel(self, model_path,
            crypto_mode = "plain_text", primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        java_model = callBigDlFunc(self.bigdl_type,
                "loadLightGBMClassificationModel",
                self.value, model_path,
                crypto_mode, primary_key_name)
        return JavaParams._from_java(java_model)

    def loadLightGBMRegressionModel(self, model_path,
            crypto_mode = "plain_text", primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        java_model = callBigDLFunc(self.bigdl_type,
                "loadLightGBMRegressionModel", model_path,
                crypto_mode, primary_key_name)
        return JavaParams._from_java(java_model)

    def loadLightGBMRankerModel(self, model_path,
            crypto_mode = "plain_text", primary_key_name = ""):
        if isinstance(crypto_mode, CryptoMode):
            crypto_mode = crypto_mode.value
        java_model = callBigDLFunc(self.bigdl_type,
                "loadLightGBMRankerModel", model_path,
                crypto_mode, primary_key_name)
        return JavaParams._from_java(java_model)

class EncryptedDataFrameReader:
    def __init__(self, bigdl_type, df_reader):
        self.bigdl_type = bigdl_type
        self.df_reader = df_reader

    def schema(self, value):
        self.df_reader = callBigDlFunc(self.bigdl_type, "schema", self.df_reader, value)
        return self

    def option(self, key, value):
        self.df_reader = callBigDlFunc(self.bigdl_type, "option", self.df_reader, key, value)
        return self

    def csv(self, path):
        return callBigDlFunc(self.bigdl_type, "csv", self.df_reader, path)

    def parquet(self, path):
        return callBigDlFunc(self.bigdl_type, "parquet", self.df_reader, path)

    def json(self, path):
        return callBigDlFunc(self.bigdl_type, "json", self.df_reader, path)


class EncryptedDataFrameWriter:
    support_mode = {"overwrite", "append", "ignore", "error", "errorifexists"}

    def __init__(self, bigdl_type, df_writer):
        self.bigdl_type = bigdl_type
        self.df_writer = df_writer

    def option(self, key, value):
        self.df_writer = callBigDlFunc(self.bigdl_type, "option", self.df_writer, key, value)
        return self

    def mode(self, mode):
        invalidInputError(mode in EncryptedDataFrameWriter.support_mode,
                          "Unknown save mode: " + mode + "." +
                          "Accepted save modes are 'overwrite', 'append', 'ignore', 'error', 'errorifexists'.")
        self.df_writer = callBigDlFunc(self.bigdl_type, "mode", self.df_writer, mode)
        return self

    def csv(self, path):
        return callBigDlFunc(self.bigdl_type, "csv", self.df_writer, path)

    def parquet(self, path):
        return callBigDlFunc(self.bigdl_type, "parquet", self.df_writer, path)

    def json(self, path):
        return callBigDlFunc(self.bigdl_type, "json", self.df_writer, path)


class CryptoMode(Enum):
    """
    BigDL crypto mode for encrypt and decrypt data.
    """

    # CryptoMode PLAIN_TEXT
    PLAIN_TEXT = "plain_text"

    # CryptoMode AES_CBC_PKCS5PADDING
    AES_CBC_PKCS5PADDING = "AES/CBC/PKCS5Padding"

    # CryptoMode AES_GCM_V1 for parquet only
    AES_GCM_V1 = "AES_GCM_V1"

    # CryptoMode AES_GCM_CTR_V1 for parquet only
    AES_GCM_CTR_V1 = "AES_GCM_CTR_V1"

def init_keys(app_id, api_key, primary_key_path):
    return callBigDlFunc("float", "initKeys", app_id, api_key, primary_key_path)


def generate_encrypted_file(kms, primary_key_path, data_key_path, input_path, output_path):
    callBigDlFunc("float", "generateEncryptedFile",
                  kms, primary_key_path, data_key_path, input_path, output_path)
