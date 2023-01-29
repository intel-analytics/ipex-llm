from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def sql_functions_f_example(spark):

    # factorial
    df = spark.createDataFrame([(5,)], ['n'])
    df.select(factorial(df.n).alias('f')).show()
    print("factorial API finished")

    # first
    df = spark.createDataFrame([(None,), (5,), (6,)], ['n'])
    df.select(first(df.n)).show()
    print("first API finished")

    # flatten
    #the result is null if there has 'None' in arrays
    df = spark.createDataFrame([([[1, 2, 3], [4, 5], [6]],), ([None, [4, 5]],)], ['data'])
    df.select(flatten(df.data).alias('r')).show()
    print("flatten API finished")

    # floor
    df = spark.createDataFrame([(1.2,), (5.9,), (6.3,)], ['n'])
    df.select(floor(df.n)).show()
    print("floor API finished")

    # format_number
    spark.createDataFrame([(5,)], ['a']).select(format_number('a', 4).alias('v')).show()
    print("format_number API finished")
    
    # format_string
    df = spark.createDataFrame([(5, "hello")], ['a', 'b'])
    df.select(format_string('%d %s', df.a, df.b).alias('v')).show()
    print("format_string API finished")

    # from_json
    data = [(1, '''{"a": 1}''')]
    schema = StructType([StructField("a", IntegerType())])
    df = spark.createDataFrame(data, ("key", "value"))
    df.select(from_json(df.value, schema).alias("json")).show()
    df.select(from_json(df.value, "MAP<STRING,INT>").alias("json")).show()
    print("from_json API finished")

    # from_unixtime
    time_df = spark.createDataFrame([(1428476400,)], ['unix_time'])
    time_df.select(from_unixtime('unix_time').alias('ts')).show()
    print("from_unixtime API finished")

    # from_utc_timestamp
    df = spark.createDataFrame([('1997-02-28 10:30:00', 'JST')], ['ts', 'tz'])
    df.select(from_utc_timestamp(df.ts, "PST").alias('local_time')).show()
    df.select(from_utc_timestamp(df.ts, df.tz).alias('local_time')).show()
    print("from_utc_timestamp API finished")
    
    print("Finish running function_f API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_f API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sql_functions_f_example(spark)
    spark.stop()
