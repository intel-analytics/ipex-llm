from pyspark.sql.functions import *
from pyspark.sql import Row, Window, SparkSession, SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min, col, mean
import random

def sql_window_api(spark):
    
    print("Start running Window and WindowSpec API")
    
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    # orderBy, partitionBy, rowsBetween, rangeBetween
    df = spark.createDataFrame([("Alice", 2, 50), ("Alice", 3, 50), ("Alice", 2, 60), ("Alice", 3, 60), ("Alice", 2, 70), ("Bob", 3, 50), ("Bob", 3, 60), ("Bob", 4, 50)], ["name", "age", "height"])
    window = Window().partitionBy("name")
    df.withColumn("mean", mean("height").over(window)).show()
    window = Window().partitionBy("name").orderBy("height").rangeBetween(-4, 0)
    df.withColumn("mean", mean("height").over(window)).show()
    window = Window().partitionBy("name").orderBy("height").rowsBetween(Window.currentRow, 1)
    df.withColumn("mean", mean("height").over(window)).show()

    print("Finish running Window and WindowSpec API")


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Window example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_window_api(spark)
