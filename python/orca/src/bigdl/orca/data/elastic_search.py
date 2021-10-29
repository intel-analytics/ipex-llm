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

from bigdl.orca import OrcaContext
from bigdl.dllib.nncontext import init_nncontext


class elastic_search:
    """
    Primary DataFrame-based loading data from elastic search interface,
    defining API to read data from ES to DataFrame.
    """

    def __init__(self):
        pass

    @staticmethod
    def read_df(esConfig, esResource, schema=None):
        """
        Read the data from elastic search into DataFrame.

        :param esConfig: Dictionary which represents configuration for
               elastic search(eg. ip, port etc).
        :param esResource: resource file in elastic search.
        :param schema: Optional. Defines the schema of Spark dataframe.
                If each column in Es is single value, don't need set schema.
        :return: Spark DataFrame. Each row represents a document in ES.
        """
        sc = init_nncontext()
        spark = OrcaContext.get_spark_session()

        reader = spark.read.format("org.elasticsearch.spark.sql")

        for key in esConfig:
            reader.option(key, esConfig[key])
        if schema:
            reader.schema(schema)

        df = reader.load(esResource)
        return df

    @staticmethod
    def flatten_df(df):
        fields = elastic_search.flatten(df.schema)
        flatten_df = df.select(fields)
        return flatten_df

    @staticmethod
    def flatten(schema, prefix=None):
        from pyspark.sql.types import StructType
        fields = []
        for field in schema.fields:
            name = prefix + '.' + field.name if prefix else field.name
            dtype = field.dataType

            if isinstance(dtype, StructType):
                fields += elastic_search.flatten(dtype, prefix=name)
            else:
                fields.append(name)
        return fields

    @staticmethod
    def write_df(esConfig, esResource, df):
        """
        Write the Spark DataFrame to elastic search.

        :param esConfig: Dictionary which represents configuration for
               elastic search(eg. ip, port etc).
        :param esResource: resource file in elastic search.
        :param df: Spark DataFrame that will be saved.
        """
        wdf = df.write.format("org.elasticsearch.spark.sql")\
            .option("es.resource", esResource)

        for key in esConfig:
            wdf.option(key, esConfig[key])

        wdf.save()

    @staticmethod
    def read_rdd(esConfig, esResource=None, filter=None, esQuery=None):
        """
        Read the data from elastic search into Spark RDD.

        :param esConfig: Dictionary which represents configuration for
               elastic search(eg. ip, port, es query etc).
        :param esResource: Optional. resource file in elastic search.
               It also can be set in esConfig
        :param filter: Optional. Request only those fields from Elasticsearch
        :param esQuery: Optional. es query
        :return: Spark RDD
        """
        sc = init_nncontext()
        if "es.resource" not in esConfig:
            esConfig["es.resource"] = esResource
        if filter is not None:
            esConfig["es.read.source.filter"] = filter
        if esQuery is not None:
            esConfig["es.query"] = esQuery
        rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
                                 "org.apache.hadoop.io.NullWritable",
                                 "org.elasticsearch.hadoop.mr.LinkedMapWritable",
                                 conf=esConfig)
        return rdd
