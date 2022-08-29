from pyspark.sql.functions import *
from pyspark.sql import Row, Window, SparkSession, SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min, col, mean
import random

def sql_dataframe_reader_api(spark):
    
    print("Start running dataframe reader API")
    
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    
    # csv
    df = spark.read.csv('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/ages.csv')
    print(df.dtypes)
    rdd = sc.textFile('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/ages.csv')
    df2 = spark.read.option('header','true').csv(rdd)
    print(df2.dtypes)
    print("csv and option API finished")

    # format 
    df = spark.read.format('json').load('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/people.json')
    print(df.dtypes)
    print("format API finished")

    # jdbc

    # json
    df1 = spark.read.json('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/people.json')
    print(df1.dtypes)
    rdd = sc.textFile('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/people.json')
    df2 = spark.read.json(rdd)
    print(df2.dtypes)
    print("json API finished")

    # load
    df = spark.read.format('json').load(['/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/people.json', '/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/people1.json'])
    print(df.dtypes)

    # orc
    df = spark.read.orc('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/orc_partitioned')
    print(df.dtypes)
    print("orc API finished")

    # parquet
    df = spark.read.parquet('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/parquet_partitioned')
    print(df.dtypes)
    print("parquet API finished")

    # schema
    s = spark.read.schema("col0 INT, col1 DOUBLE")
    print(s)
    print("schema API finished")

    # table
    df = spark.read.parquet('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/parquet_partitioned')
    df.createOrReplaceTempView('tmpTable')
    res = spark.read.table('tmpTable').dtypes
    print(res)
    print("table API finished")

    # text
    df = spark.read.text('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/text-test.txt')
    df.show()
    df = spark.read.text('/ppml/trusted-big-data-ml/work/spark-3.1.2/python/test_support/sql/text-test.txt', wholetext=True)
    df.show()
    print("text API finished")

    print("Finish running dataframe reader API")


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Dataframe Reader example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_dataframe_reader_api(spark)
