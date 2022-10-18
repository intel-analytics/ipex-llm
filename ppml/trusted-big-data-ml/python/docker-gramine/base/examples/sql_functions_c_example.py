from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def sql_functions_c_example(spark):
    
    # cbrt
    #return cube-root value
    df = spark.createDataFrame([(1,), (2,), (3,), (8,)], ["n1"])
    df.select(cbrt(df.n1).alias("cube-root_value")).show()
    print("cbrt API finished")

    # ceil
    df = spark.createDataFrame([(1.1,), (2.6,), (3.3,), (8.7,)], ["n1"])
    df.select(ceil(df.n1).alias("cube-root_value")).show()
    print("ceil API finished")

    # coalesce
    cDf = spark.createDataFrame([(None, None), (1, None), (None, 2)], ("a", "b"))
    cDf.show()
    cDf.select(coalesce(cDf["a"], cDf["b"])).show()
    cDf.select('*', coalesce(cDf["a"], lit(0.0))).show()
    print("coalesce API finished")

    # col
    df = spark.createDataFrame([('1', '3'), ('8', '9'), ('12', '24')], ["n1", "n2"])
    df.select(col("n1")).show()
    print("col API finished")

    # collect_list
    df2 = spark.createDataFrame([(2,), (5,), (5,)], ('age',))
    df2.agg(collect_list('age')).show()
    print("collect_list API finished")

    # collect_set
    df2 = spark.createDataFrame([(2,), (5,), (5,)], ('age',))
    df2.agg(collect_set('age')).show()
    print("collect_set API finished")

    # column
    #There is bug in the source code
    # df = spark.createDataFrame([('1', '3'), ('8', '9'), ('12', '24')], ["n1", "n2"])
    # df.select(column("n1")).show()

    # concat
    df = spark.createDataFrame([([1, 2, 3, 4], [2, 3, 4, 5])], ['s', 'd'])
    df.select(concat(df.s, df.d).alias('s')).show()
    df = spark.createDataFrame([('abcd','123')], ['s', 'd'])
    df.select(concat(df.s, df.d).alias('s')).show()
    print("concat API finished")

    # concat_ws
    df = spark.createDataFrame([('abcd','123')], ['s', 'd'])
    df.select(concat_ws('-', df.s, df.d).alias('s')).show()
    print("concat_ws API finished")

    # conv
    df = spark.createDataFrame([("010101",)], ['n'])
    df.select(conv(df.n, 2, 16).alias('hex')).show()
    print("conv API finished")

    # corr
    a = range(20)
    b = [2 * x for x in range(20)]
    df = spark.createDataFrame(zip(a, b), ["a", "b"])
    df.agg(corr("a", "b").alias('c')).collect()
    print("corr API finished")

    # cos
    df = spark.createDataFrame([(30,), (60,)], ['n1'])
    df.select(cos(df.n1)).show()
    print("cos API finished")

    # cosh
    df = spark.createDataFrame([(30,), (60,)], ['n1'])
    df.select(cosh(df.n1)).show()
    print("cosh API finished")

    # count
    df = spark.createDataFrame([('1', '3'), ('8', '9'), ('12', '24')], ["n1", "n2"])
    df.select(count(df.n1)).show()
    print("count API finished")

    # countDistinct
    df = spark.createDataFrame([('1', '3'), ('1', '3'), ('1', '4')], ["n1", "n2"])
    df.select(countDistinct(df.n1, df.n2)).show()
    df = spark.createDataFrame([('1', '2'), ('1', '3'), ('1', '4')], ["n1", "n2"])
    df.select(countDistinct(df.n1, df.n2)).show()
    print("countDistinct API finished")

    # covar_pop
    a = [1] * 10
    b = [2] * 10
    df = spark.createDataFrame(zip(a, b), ["a", "b"])
    df.agg(covar_pop("a", "b").alias('c')).show()
    print("covar_pop API finished")

    # covar_samp
    a = [1] * 10
    b = [1] * 10
    df = spark.createDataFrame(zip(a, b), ["a", "b"])
    df.agg(covar_samp("a", "b").alias('c')).collect()
    print("covar_samp API finished")

    # crc32
    spark.createDataFrame([('ABC',)], ['a']).select(crc32('a').alias('crc32')).show()
    print("crc32 API finished")

    # create_map
    df = spark.createDataFrame([('Alice', 20), ('Bob', 22)], ['name', 'age'])
    df.select(create_map('name', 'age').alias('map')).show()
    print("create_map API finished")

    # cume_dist
    from pyspark.sql import Window
    # need orderBy clause.
    window = Window.orderBy("name")
    df.withColumn("cumulative_distribution", cume_dist().over(window)).show()

    # current_date
    df = spark.createDataFrame([('Alice', 20), ('Bob', 22)], ['name', 'age'])
    df.select(current_date().alias('date')).show()
    print("current_date API finished")

    # current_timestamp
    df = spark.createDataFrame([('Alice', 20), ('Bob', 22)], ['name', 'age'])
    df.select(current_timestamp().alias('timestamp')).show()
    print("current_timestamp API finished")
    
    print("Finish running function_c API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_c API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sql_functions_c_example(spark)
    spark.stop()
