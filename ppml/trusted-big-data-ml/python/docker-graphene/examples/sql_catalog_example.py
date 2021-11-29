from pyspark.sql import SparkSession, Catalog

def sql_catalog_example(spark):
    
    # create catalog object
    catalog = Catalog(spark)
    
    # createTable 
    table_df = catalog.createTable("table", "/ppml/trusted-big-data-ml/work/spark-2.4.6/examples/src/main/resources/people.txt", "text")
    table_df.show()
    print("createTable API finished")

    # currentDatabase
    cur_db = catalog.currentDatabase()
    print("current database: {}".format(cur_db))
    print("currentDatabase API finished")

    # createTable
    catalog.createTable("table3", "/ppml/trusted-big-data-ml/work/spark-2.4.6/examples/src/main/resources/people.txt", "text")
    spark.sql("select * from table3").show()
    print("createTable API finished")
    
    # listColumns
    catalog.listColumns("table3")
    print("listColumns API finished")

    # listTables
    catalog.listTables()
    catalog.listTables(dbName="default")
    print("listTables API finished")

    # dropTempView
    spark.createDataFrame([(1, 1)]).createTempView("my_table")
    spark.table("my_table").show()
    catalog.dropTempView("my_table")
    print("dropTempView API finished")

    # dropGlobalTempView
    spark.createDataFrame([(1, 1)]).createGlobalTempView("my_table")
    spark.table("global_temp.my_table").show()
    catalog.dropGlobalTempView("my_table")
    print("dropGlobalTempView API finished")

    # cacheTable & isCached
    print("table3 is cached: {}".format(catalog.isCached("table3")))
    catalog.cacheTable("table3")
    print("table3 is cached: {}".format(catalog.isCached("table3")))
    print("cacheTable & isCached API finished")
    
    # uncacheTable
    catalog.uncacheTable("table3")
    print("table3 is cached: {}".format(catalog.isCached("table3")))
    print("uncacheTable API finished")

    # clearCache
    catalog.cacheTable("table")
    catalog.cacheTable("table3")
    catalog.clearCache()
    print("table is cached: {}".format(catalog.isCached("table")))
    print("table3 is cached: {}".format(catalog.isCached("table3")))
    print("clearCache API finished")

    # listFunctions
    funs = catalog.listFunctions()
    funs1 = catalog.listFunctions("default")
    print("there are {} functions registed on this db".format(funs1))
    print("listFunctions API finished")

    # createExternalTable
    # deprecated

    # recoverPartitions
    # TODO

    # refreshByPath
    # 

    # refreshTable

    # registerFunction
    # Deprecated

    # setCurrentDatabase
    catalog.setCurrentDatabase("default")
    print("setCurrentDatabase API finished")
    
    spark.stop()

    print("Finish running Catalog API")

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Catalog example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    
    sql_catalog_example(spark)
