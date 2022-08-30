from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def sql_functions_b_example(spark):
    
    # base64
    #encoding a binary column
    df = spark.createDataFrame([('1',), ('2',), ('10',)], ["n1"])
    df.withColumn("base64_n1", base64(df.n1)).show()
    print("base64 API finished")

    # bin
    #arg type: bigint type
    #return: the string representation of the binary
    df = spark.createDataFrame([(1,), (2,), (3,)], ["n1"])
    df.select(bin(df.n1).alias("binary_number")).show()
    print("bin API finished")

    # bitwiseNOT
    #arg type: integral type
    #return: bitwise not value
    df = spark.createDataFrame([(1,), (2,), (3,)], ["n1"])
    df.select(bitwiseNOT(df.n1).alias("bitwise_not_value")).show()
    print("bitwiseNOT API finished")

    # broadcast
    #assume df1 (few KB) << df2 (10s of GB)
    df1 = spark.createDataFrame([(1, 'aa'), (4, 'dd')], ["n1", "s1"])
    df2 = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c'), (5, 'e'), (6, 'f')], ["n2", "s2"])
    df1.join(broadcast(df2), df1.n1 == df2.n2).show()
    print("broadcast API finished")

    # bround
    spark.createDataFrame([(2.5,)], ['a']).select(bround('a', 0).alias('r')).collect()
    print("bround API finished")

    print("Finish running function_b API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL functions_b API example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_functions_b_example(spark)
    spark.stop()
