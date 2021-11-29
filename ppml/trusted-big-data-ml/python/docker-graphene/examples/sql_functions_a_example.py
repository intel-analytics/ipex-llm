from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *

def sql_functions_a_example(spark):
    
    # abs
    df = spark.createDataFrame([("account1", -100),("accout2", 360)], ("account", "value"))
    df.withColumn("abs_num", abs(df.value)).show()
    print("abs API finished")

    # acos & asin & atan
    df = spark.createDataFrame([("num1", 1.0),("num2", 0.5)], ("number", "value"))
    df.withColumn("acos", acos(df.value)).show()
    df.withColumn("asin", asin(df.value)).show()
    df.withColumn("atan", atan(df.value)).show()
    print("acos & asin & atan API finished")

    # add_months
    df = spark.createDataFrame([('2015-04-08',)], ['dt'])
    df.select(add_months(df.dt, 1).alias('next_month')).show()
    print("add_months API finished")

    # approx_count_distinct
    df = spark.createDataFrame([("num1", 1.0),("num2", 2.0),("num3", 8.0), ("num4", 0.0)], ("number", "value"))
    df.select(approx_count_distinct(df.number)).show()
    print("approx_count_distinct API finished")

    # array
    df = spark.createDataFrame([("num1", 1.0),("num2", 2.0),("num3", 8.0), ("num4", 0.0)], ("number", "value"))
    df.select(array('number', 'value')).show()
    print("array API finished")

    # array_contains
    df = spark.createDataFrame([("num1", 1.0),("num2", 2.0),("num3", 8.0), ("num4", 0.0)], ("number", "value"))
    df = df.withColumn("array_column", array("number", "value"))
    df.select(array_contains(df.array_column, "num1")).show()
    print("array_contains API finished")

    # array_distinct
    df = spark.createDataFrame([([1, 2, 3, 2],), ([4, 5, 5, 4],)], ['data'])
    df.select(array_distinct(df.data)).show()
    print("array_distinct API finished")

    # array_except
    df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2=["c", "d", "a", "f"])])
    df.select(array_except(df.c1, df.c2)).show()
    df.select(array_except(df.c2, df.c1)).show()
    print("array_except API finished")

    # array_intersect
    df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2=["c", "d", "a", "f"])])
    df.select(array_intersect(df.c1, df.c2)).show()
    print("array_intersect API finished")

    # array_join
    df = spark.createDataFrame([(["a", "b", "c"],), (["a", None],)], ['data'])
    df.select(array_join(df.data, ",").alias("joined")).show()
    df.select(array_join(df.data, ",", "NULL").alias("joined")).show()
    print("array_join API finished")

    # array_max & array_min
    df = spark.createDataFrame([([1, 2, 3, 4, 5, 6, 7, 8], ), ([-1, -2, -3], )], ["data"])
    df.select(array_max(df.data)).show()
    df.select(array_min(df.data)).show()
    print("array_max & array_min API finished")

    # array_position
    df = spark.createDataFrame([(["c", "b", "a"],), ([],)], ['data'])
    df.select(array_position(df.data, "a")).collect()
    print("array_position API finished")

    # array_remove
    df = spark.createDataFrame([([1, 2, 3, 1, 1],), ([],)], ['data'])
    df.select(array_remove(df.data, 1)).show()
    print("array_remove API finished")
    
    # array_repeat
    df = spark.createDataFrame([('ab',), ('cd',)], ['data'])
    df.select(array_repeat(df.data, 3)).show()
    print("array_repeat API finished")

    # array_sort
    df = spark.createDataFrame([([2, 1, None, 3],),([1],),([],)], ['data'])
    df.select(array_sort(df.data).alias('r')).show()
    print("array_sort API finished")

    # array_union
    df = spark.createDataFrame([Row(c1=["b", "a", "c"], c2=["c", "d", "a", "f"])])
    df.select(array_union(df.c1, df.c2)).show()
    print("array_union API finished")

    # arrays_overlap
    df = spark.createDataFrame([(["a", "b"], ["b", "c"]), (["a"], ["b", "c"])], ['x', 'y'])
    df.select(arrays_overlap(df.x, df.y).alias("overlap")).show()
    # return: [Row(overlap=True), Row(overlap=None)]
    df = spark.createDataFrame([(["a", "b"], ["b", "c"]), ([None], ["b", "c"])], ['x', 'y'])
    df.select(arrays_overlap(df.x, df.y).alias("overlap")).show()
    # return: [Row(overlap=True), Row(overlap=True)]
    df = spark.createDataFrame([(["a", "b"], ["b", "c"]), ([None, "b"], ["b", "c"])], ['x', 'y'])
    df.select(arrays_overlap(df.x, df.y).alias("overlap")).show()
    print("arrays_overlap API finished")

    # arrays_zip
    df = spark.createDataFrame([([1, 2, 3], [2, 3, 4])], ['vals1', 'vals2'])
    # >>> df.select(arrays_zip(df.vals1, df.vals2).alias('zipped'))
    #  DataFrame[zipped: array<struct<vals1:bigint,vals2:bigint>>]
    df.select(arrays_zip(df.vals1, df.vals2).alias('zipped')).show()
    print("arrays_zip API finished")

    # asc
    # return a sort expression
    df = spark.createDataFrame([(1,), (10,), (8,), (7,), (5,), (4,)], ["n1"])
    print("asc type: ", type(df.n1.asc()))
    df.sort(df.n1.asc()).show()
    print("asc API finished")

    # asc_nulls_first & asc_nulls_last
    df = spark.createDataFrame([(1,), (10,), (8,), (7,), (5,), (4,), (None,)], ["n1"])
    df.sort(df.n1.asc_nulls_first()).show()
    df.sort(df.n1.asc_nulls_last()).show()
    print("asc_nulls_first & asc_nulls_last API finished")

    # ascii
    df = spark.createDataFrame([('a', 1), ('b', 10), ('c', 12)], ["string", "integer"])
    df.select(ascii(df.string)).show()
    print("ascii API finished")

    # atan2
    df = spark.createDataFrame([(1.0, 1.0), (1.0, 0.5), (0.5, 1.0)], ["y", "x"])
    df.withColumn("atan2", atan2(df.y, df.x)).show()
    print("atan2 API finished")

    # avg
    df = spark.createDataFrame([(1,), (10,), (8,), (7,), (5,), (4,)], ["n1"])
    df.select(avg(df.n1).alias("avg")).show()
    print("avg API finished")
    
    print("Finish running function_a API")

if __name__ == "__main__":
    
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_a API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_functions_a_example(spark)
    spark.stop()
