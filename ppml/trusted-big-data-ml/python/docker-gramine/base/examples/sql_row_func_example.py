from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min
import random

def sql_row_func_api(spark):
    
    print("Start running Row and Functions API")
    
    # row
    row = Row(name="Alice", age=11)
    print(row)
    print(row.name, row.age)
    Person = Row("name", "age")
    print(Person)
    print(Person("Alice", 11))
    print("row API finished")
    
    # asDict
    row = Row(key=1, value=Row(name='a', age=2))
    res = (row.asDict() == {'key': 1, 'value': Row(age=2, name='a')})
    print(res)
    print("asDict API finished")
    
    # drop and fill
    df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    df.na.drop().show()
    df.na.fill(50).show()
    print("drop and fill API finished")
    
    # replace
    df = spark.createDataFrame([('Tom', 80), (None, 60), ('Alice', None)], ["name", "height"])
    df.na.replace('Alice', None).show()
    print("replace API finished")
    
    print("Finish running SQL Row_and_DataFrameNaFunctions API")




if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Row and Functions example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_row_func_api(spark)
