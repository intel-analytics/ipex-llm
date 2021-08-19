from pyspark.sql.functions import *
from pyspark.sql import Row, Window, SparkSession, SQLContext
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min, col, mean
import random
import os
import tempfile

def sql_dataframe_writer_api(spark):
    
    print("Start running dataframe writer API")
    
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    
    # bucketBy and saveAsTable
    # make sure that this is no work/spark-warehouse/bucketed_table/ under current path
    df.write.format('parquet').bucketBy(100, 'age', 'name').mode("overwrite").saveAsTable('bucketed_table', path="work/spark-warehouse/bucketed_table/")
    print("bucketBy and saveAsTable API finished")

    # csv and option
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df.write.option('header','true').csv(os.path.join(tempfile.mkdtemp(), 'data'))
    print("csv and option API finished")

    # format
    df.write.format('json').save(os.path.join(tempfile.mkdtemp(), 'data'))
    print("format API finished")

    # insertInto
    df2 =  spark.createDataFrame([(3, "Alice")], ["age", "name"])
    df2.write.insertInto("bucketed_table")
    print("insertInto API finished")
    
    # jdbc

    # json
    df.write.json(os.path.join(tempfile.mkdtemp(), 'data'))
    print("json API finished")

    # mode
    df.write.mode('append').parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    print("mode API finished")
    
    # orc
    orc_df = spark.read.orc('/ppml/trusted-big-data-ml/work/spark-2.4.6/python/test_support/sql/orc_partitioned')
    orc_df.write.orc(os.path.join(tempfile.mkdtemp(), 'data'))
    print("orc API finished")

    # parquet
    df.write.parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    print("parquet API finished")
    
    # partitionBy
    df.write.partitionBy('age').parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    print("partitionBy API finished")

    # save
    df.write.mode("append").save(os.path.join(tempfile.mkdtemp(), 'data'))
    print("save API finished")

    # sortBy
    # make sure that this is no work/spark-warehouse/sorted_bucketed_table/ under current path
    df.write.format('parquet').bucketBy(100, 'name').sortBy('age').mode("overwrite").saveAsTable('sorted_bucketed_table', path="work/spark-warehouse/sorted_bucketed_table/")
    print("sortBy API finished")
    
    # text
    df = spark.createDataFrame([1.0, 2.0, 3.0], StringType())
    df.write.text(os.path.join(tempfile.mkdtemp(), 'data'))
    print("text API finished")

    print("Finish running dataframe writer API")


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Dataframe Writer example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_dataframe_writer_api(spark)
