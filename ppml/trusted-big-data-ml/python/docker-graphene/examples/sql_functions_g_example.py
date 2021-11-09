from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def sql_functions_g_example(spark):
    
    # get_json_object
    data = [("1", '''{"f1": "value1", "f2": "value2"}'''), ("2", '''{"f1": "value12"}''')]
    df = spark.createDataFrame(data, ("key", "jstring"))
    df.select(df.key, get_json_object(df.jstring, '$.f1').alias("c0"), \
                      get_json_object(df.jstring, '$.f2').alias("c1") ).show()
    print("get_json_object API finished")
    
    # greatest
    df = spark.createDataFrame([(1, 4, 3), (5, 1, 2)], ['a', 'b', 'c'])
    df.select(greatest(df.a, df.b, df.c).alias("greatest")).show()
    print("greatest API finished")

    # grouping
    df = spark.createDataFrame([('Boc', 22), (None, 20), ('Alice', 21), ('Alice', None)], ("name", "age"))
    df.cube('name').agg(grouping("name"), sum("age")).orderBy("name").show()
    print("grouping API finished")

    # grouping_id
    df = spark.createDataFrame([('Boc', 22), (None, 20), ('Alice', 21), ('Alice', None)], ("name", "age"))
    df.cube("name").agg(grouping_id(), sum("age")).orderBy("name").show()
    print("grouping_id API finished")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_g API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sql_functions_g_example(spark)
    spark.stop()
