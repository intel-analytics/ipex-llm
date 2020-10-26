#
# Copyright 2018 Analytics Zoo Authors.
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
from pyspark.sql.types import *
from pyspark.sql import SQLContext

from zoo.common.nncontext import init_nncontext


class elastic_search:
    """
    Primary DataFrame-based loading data from elastic search interface,
    defining API to read data from ES to DataFrame.
    """

    def __init__(self):
        pass

    @staticmethod
    def read(esConfig, esResource, schema=None):
        """
        Read the data from elastic search into DataFrame.
        :param esConfig Dictionary which represents configuration for
               elastic search(eg. ip, port etc).
        :param esResource resource file in elastic search.
        :param schema Optional. Defines the schema of Spark dataframe.
                If each column in Es is single value, don't need set schema.
        :return Spark DataFrame. Each row represents a document in ES.
        """
        sc = init_nncontext()
        sqlContext = SQLContext.getOrCreate(sc)
        spark = sqlContext.sparkSession

        reader = spark.read.format("org.elasticsearch.spark.sql")

        for key in esConfig:
            reader.option(key, esConfig[key])
        if schema:
            reader.schema(schema)

        df = reader.load(esResource)
        return df

    @staticmethod
    def write(esConfig, esResource, df):
        """
        Read the data from elastic search into DataFrame.
        :param esConfig Dictionary which represents configuration for
               elastic search(eg. ip, port etc).
        :param esResource resource file in elastic search.
        :param df Spark DataFrame that will be saved.
        """
        wdf = df.write.format("org.elasticsearch.spark.sql")\
            .option("es.resource", esResource)

        for key in esConfig:
            wdf.option(key, esConfig[key])

        wdf.save()
