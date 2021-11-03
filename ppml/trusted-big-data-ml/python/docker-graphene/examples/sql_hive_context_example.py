from pyspark.sql import SparkSession, HiveContext

def sql_hive_context_example(spark):
    
    # create hive context object.
    hive_ctx = HiveContext(spark.sparkContext)

    # createDataFrame
    l = [('Alice', 18), ('Bob', 20), ('Charley', 22)]
    df = hive_ctx.createDataFrame(l, ('name', 'age'))
    print("createDataFrame API finished")

    # registerDataFrameAsTable 
    hive_ctx.registerDataFrameAsTable(df, "table1")
    print("registerDataFrameAsTable API finished")

    # sql
    tmp_df = hive_ctx.sql("select * from table1")
    tmp_df.show()
    print("sql API finished")

    # table
    tmp_df = hive_ctx.table("table1")
    tmp_df.show()
    print("table API finished")

    # tableNames
    table_names = hive_ctx.tableNames()
    print(table_names)
    print("tableNames API finished")

    # tables
    tables = hive_ctx.tables()
    print(tables)
    print("tables API finished")

    # range
    tmp_df = hive_ctx.range(1,10,2)
    tmp_df.show()
    print("range API finished")

    # dropTempTable
    hive_ctx.dropTempTable("table1")
    table_names = hive_ctx.tableNames()
    print(table_names)
    print("dropTempTable API finished")

    # cacheTable & uncacheTable & clearCache
    df = hive_ctx.range(1,10,2)
    hive_ctx.registerDataFrameAsTable(df, "table")
    hive_ctx.cacheTable("table")
    hive_ctx.uncacheTable("table")
    hive_ctx.clearCache()
    print("cacheTable & uncacheTable & clearCache API finished")

    # createExternalTable

    # newSession

    # registerFunction
    # Deprecated in 2.3.0. Use :func:`spark.udf.register` instead

    # registerJavaFunction
    # Deprecated in 2.3.0. Use :func:`spark.udf.registerJavaFunction` instead

    # setConf & getConf
    hive_ctx.setConf("key1", "value1")
    value = hive_ctx.getConf("key1")
    print(value)
    print("setConf & getConf API finished")

    # refreshTable
    # Exception: An error occurred while calling o26.refreshTable:
    # Method refreshTable([class java.lang.String]) does not exist


    

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .enableHiveSupport() \
        .appName("Python Spark SQL Hive Context example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_hive_context_example(spark)

