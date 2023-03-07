#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

class TpchBase:
    def __init__(self, spark, dir):
        # define schema
        lineitem_schema = StructType([
            StructField("l_orderkey", IntegerType()),
            StructField("l_partkey", IntegerType()),
            StructField("l_suppkey", IntegerType()),
            StructField("l_linenumber", IntegerType()),
            StructField("l_quantity", DoubleType()),
            StructField("l_extendedprice", DoubleType()),
            StructField("l_discount", DoubleType()),
            StructField("l_tax", DoubleType()),
            StructField("l_returnflag", StringType()),
            StructField("l_linestatus", StringType()),
            StructField("l_shipdate", StringType()),
            StructField("l_commitdate", StringType()),
            StructField("l_receiptdate", StringType()),
            StructField("l_shipinstruct", StringType()),
            StructField("l_shipmode", StringType()),
            StructField("l_comment", StringType())
        ])
        self.lineitem = spark.read.format("csv") \
            .option("delimiter", "|") \
            .option("header", "false") \
            .schema(lineitem_schema) \
            .load(dir + "./lineitem.tbl")

        nation_schema = StructType([
            StructField("n_nationkey", IntegerType(), True),
            StructField("n_name", StringType(), True),
            StructField("n_regionkey", IntegerType(), True),
            StructField("n_comment", StringType(), True)
        ])
        self.nation = spark.read.format("csv") \
            .option("sep", "|") \
            .option("header", "false") \
            .schema(nation_schema) \
            .load(dir + "./nation.tbl")

        order_schema = StructType([
            StructField("o_orderkey", IntegerType(), True),
            StructField("o_custkey", IntegerType(), True),
            StructField("o_orderstatus", StringType(), True),
            StructField("o_totalprice", DoubleType(), True),
            StructField("o_orderdate", StringType(), True),
            StructField("o_orderpriority", StringType(), True),
            StructField("o_clerk", StringType(), True),
            StructField("o_shippriority", IntegerType(), True),
            StructField("o_comment", StringType(), True)
        ])
        self.orders = spark.read.format("csv") \
            .option("sep", "|") \
            .option("header", "false") \
            .schema(order_schema) \
            .load(dir + "./orders.tbl")

        region_schema = StructType([
            StructField("r_regionkey", IntegerType(), False),
            StructField("r_name", StringType(), False),
            StructField("r_comment", StringType(), False)
        ])

        self.region = spark.read.format("csv") \
            .schema(region_schema) \
            .option("delimiter", "|") \
            .option("header", "false") \
            .load(dir + "./region.tbl")

        part_schema = StructType([
            StructField("p_partkey", IntegerType(), False),
            StructField("p_name", StringType(), False),
            StructField("p_mfgr", StringType(), False),
            StructField("p_brand", StringType(), False),
            StructField("p_type", StringType(), False),
            StructField("p_size", IntegerType(), False),
            StructField("p_container", StringType(), False),
            StructField("p_retailprice", DoubleType(), False),
            StructField("p_comment", StringType(), False)
        ])

        self.part = spark.read.format("csv") \
            .schema(part_schema) \
            .option("header", "false") \
            .option("delimiter", "|") \
            .load(dir + "./part.tbl")

        customer_schema = StructType([
            StructField("c_custkey", IntegerType(), False),
            StructField("c_name", StringType(), False),
            StructField("c_address", StringType(), False),
            StructField("c_nationkey", IntegerType(), False),
            StructField("c_phone", StringType(), False),
            StructField("c_acctbal", DoubleType(), False),
            StructField("c_mktsegment", StringType(), False),
            StructField("c_comment", StringType(), False)
        ])

        self.customer = spark.read.format("csv") \
            .schema(customer_schema) \
            .option("delimiter", "|") \
            .option("header", "false") \
            .load(dir + "./customer.tbl")

        supplier_schema = StructType([
            StructField("s_suppkey", IntegerType(), True),
            StructField("s_name", StringType(), True),
            StructField("s_address", StringType(), True),
            StructField("s_nationkey", IntegerType(), True),
            StructField("s_phone", StringType(), True),
            StructField("s_acctbal", DoubleType(), True),
            StructField("s_comment", StringType(), True)
        ])

        self.supplier = spark.read.format("csv") \
            .option("delimiter", "|") \
            .option("header", "false") \
            .schema(supplier_schema) \
            .load(dir + "./supplier.tbl")

        partsupp_schema = StructType([
            StructField("ps_partkey", IntegerType(), True),
            StructField("ps_suppkey", IntegerType(), True),
            StructField("ps_availqty", IntegerType(), True),
            StructField("ps_supplycost", DoubleType(), True),
            StructField("ps_comment", StringType(), True)
        ])

        self.partsupp = spark.read.format("csv") \
            .option("delimiter", "|") \
            .option("header", "false") \
            .schema(partsupp_schema) \
            .load(dir + "./partsupp.tbl")
