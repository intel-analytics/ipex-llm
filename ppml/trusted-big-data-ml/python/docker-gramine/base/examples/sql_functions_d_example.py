from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def sql_functions_d_example(spark):
    
    # date_add
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(date_add(df.dt,1).alias("next_day")).show()
    print("date_add API finished")

    # date_format
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(date_format('dt', 'MM/dd/yyy').alias('date')).show()
    print("date_format API finished")

    # date_sub
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(date_sub(df.dt, 1).alias('prev_date')).show()
    print("date_sub API finished")

    # date_trunc
    df = spark.createDataFrame([('1997-02-28 05:02:11',)], ['t'])
    df.select(date_trunc('year', df.t).alias('year')).show()
    df.select(date_trunc('mon', df.t).alias('month')).show()
    print("date_trunc API finished")

    # datediff
    #arg1:end time / arg2:start time
    #Returns the number of days from start to end
    df = spark.createDataFrame([('2015-04-08','2015-05-10')], ['d1', 'd2'])
    df.select(datediff(df.d2, df.d1).alias('diff')).show()
    print("datediff API finished")

    # dayofmonth
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(dayofmonth('dt').alias('day')).show()
    print("dayofmonth API finished")

    # dayofweek
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(dayofweek('dt').alias('day')).show()
    print("dayofweek API finished")

    # dayofyear
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(dayofyear('dt').alias('day')).show()
    print("dayofyear API finished")

    # decode
    df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ["n1", "s1"])
    df.withColumn("encode", encode(df.s1, "utf-8")).withColumn("decode", decode("encode", "utf-8")).show()
    print("decode API finished")

    # degrees
    import math
    df = spark.createDataFrame([(math.pi,), (math.pi / 6,)], ["radians"])
    df.select(degrees(df.radians)).show()
    print("degrees API finished")

    # dense_rank
    from pyspark.sql import Window
    window = Window.orderBy("score")
    df = spark.createDataFrame([('Bob', 90), ('Alice', 95), ('Coris', 90), ('David', 89)], ["name", "score"])
    df.withColumn("dense_rank", dense_rank().over(window)).show()
    df.withColumn("rank", rank().over(window)).show()
    print("dense_rank API finished")

    # desc
    df = spark.createDataFrame([(1, None), (10, 12), (8, 3), (None, 9), (9, 6)], ["n1", "n2"])
    df.sort(df.n1.desc()).show()
    print("desc API finished")

    # desc_nulls_first
    df = spark.createDataFrame([(1, None), (10, 12), (8, 3), (None, 9), (9, 6)], ["n1", "n2"])
    df.sort(df.n1.desc_nulls_first()).show()
    print("desc_nulls_first API finished")

    # desc_nulls_last
    df = spark.createDataFrame([(1, None), (10, 12), (8, 3), (None, 9), (9, 6)], ["n1", "n2"])
    df.sort(df.n1.desc_nulls_last()).show()
    print("desc_nulls_last API finished")
    
    print("Finish running function_d API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_d API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sql_functions_d_example(spark)
    spark.stop()
