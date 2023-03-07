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

import re

from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from tpch_base import TpchBase


class TpchFunctionalQueries(TpchBase):
    def __init__(self, spark, dir):
        TpchBase.__init__(self, spark, dir)

    def q1(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())
        increase = udf(lambda x, y: x * (1 + y), FloatType())
        #F.sum(increase(decrease(col("l_extendedprice"), col("l_discount")), col("l_tax"))).alias("sum_charge") is not support

        return self.lineitem.filter(col("l_shipdate") <= "1998-09-02") \
            .groupBy(col("l_returnflag"), col("l_linestatus")) \
            .agg(F.sum(col("l_quantity")).alias("sum_qty"),
                 F.sum(col("l_extendedprice")).alias("sum_base_price"),
                 F.sum(decrease(col("l_extendedprice"), col("l_discount"))).alias("sum_disc_price"),
                 F.sum(increase((1 - self.lineitem.l_discount) * self.lineitem.l_extendedprice , col("l_tax"))),
                 F.avg(col("l_quantity")).alias("avg_qty"),
                 F.avg(col("l_extendedprice")).alias("avg_price"),
                 F.avg(col("l_discount")).alias("avg_disc"),
                 F.count(col("l_quantity")).alias("count_order")) \
            .sort(col("l_returnflag"), col("l_linestatus"))

    def q2(self):
        europe = self.region.filter(col("r_name") == "EUROPE") \
            .join(self.nation, col("r_regionkey") == col("n_regionkey")) \
            .join(self.supplier, col("n_nationkey") == col("s_nationkey")) \
            .join(self.partsupp, self.supplier.s_suppkey == self.partsupp.ps_suppkey)

        brass = self.part.filter((col("p_size") == 15) & (self.part.p_type.endswith("BRASS"))) \
            .join(europe, col("p_partkey") == europe.ps_partkey)

        minimumCost = brass.groupBy(col("ps_partkey")) \
            .agg(F.min(col("ps_supplycost")).alias("min"))

        return brass.join(minimumCost, brass.ps_partkey == minimumCost.ps_partkey) \
            .filter(brass.ps_supplycost == minimumCost.min) \
            .select("s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment") \
            .sort(col("s_acctbal").desc(), col("n_name"), col("s_name"), col("p_partkey")).limit(100)

    def q3(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        filteredCustomers = self.customer.filter(col("c_mktsegment") == "BUILDING")
        filteredOrders = self.orders.filter(col("o_orderdate") < "1995-03-15")
        filteredLineItems = self.lineitem.filter(col("l_shipdate") > "1995-03-15")

        return filteredCustomers.join(filteredOrders, col("c_custkey") == col("o_custkey")) \
            .select("o_orderkey", "o_orderdate", "o_shippriority") \
            .join(filteredLineItems, col("o_orderkey") == col("l_orderkey")) \
            .select(col("l_orderkey"),
                    decrease(col("l_extendedprice"), col("l_discount")).alias("volume"),
                    col("o_orderdate"), col("o_shippriority")) \
            .groupBy(col("l_orderkey"), col("o_orderdate"), col("o_shippriority")) \
            .agg(F.sum(col("volume")).alias("revenue")) \
            .sort(col("revenue").desc(), col("o_orderdate")).limit(10)
    def q4(self):
        filteredOrders = self.orders.filter((col("o_orderdate") >= "1993-07-01") & (col("o_orderdate") < "1993-10-01"))

        filteredLineItems = self.lineitem.filter(col("l_commitdate") < col("l_receiptdate")) \
            .select("l_orderkey") \
            .distinct()

        return filteredLineItems.join(filteredOrders, col("l_orderkey") == col("o_orderkey")) \
            .groupBy("o_orderpriority") \
            .agg(F.count(col("o_orderpriority")).alias("order_count")) \
            .sort(col("o_orderpriority"))
    def q5(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        filteredOrders = self.orders.filter((col("o_orderdate") < "1995-01-01") & (col("o_orderdate") >= "1994-01-01"))

        return self.region.filter(col("r_name") == "ASIA") \
            .join(self.nation, col("r_regionkey") == col("n_regionkey")) \
            .join(self.supplier, col("n_nationkey") == col("s_nationkey")) \
            .join(self.lineitem, col("s_suppkey") == col("l_suppkey")) \
            .select("n_name", "l_extendedprice", "l_discount", "l_orderkey", "s_nationkey") \
            .join(filteredOrders, col("l_orderkey") == col("o_orderkey")) \
            .join(self.customer, (col("o_custkey") == col("c_custkey")) & (col("s_nationkey") == col("c_nationkey"))) \
            .select(col("n_name"), decrease(col("l_extendedprice"), col("l_discount")).alias("value")) \
            .groupBy("n_name") \
            .agg(F.sum(col("value")).alias("revenue")) \
            .sort(col("revenue").desc())

    def q6(self):
        return self.lineitem.filter((col("l_shipdate") >= "1994-01-01")
                             & (col("l_shipdate") < "1995-01-01")
                             & (col("l_discount") >= 0.05)
                             & (col("l_discount") <= 0.07)
                             & (col("l_quantity") < 24)) \
            .agg(F.sum(col("l_extendedprice") * col("l_discount")).alias("revenue"))

    def q7(self):
        getYear = udf(lambda x: x[0:4], StringType())
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        filteredNations = self.nation.filter((col("n_name") == "FRANCE") | (col("n_name") == "GERMANY"))

        filteredLineitems = self.lineitem.filter(
            (col("l_shipdate") >= "1995-01-01") & (col("l_shipdate") <= "1996-12-31"))

        supplierNations = filteredNations.join(self.supplier, col("n_nationkey") == col("s_nationkey")) \
            .join(filteredLineitems, col("s_suppkey") == col("l_suppkey")) \
            .select(col("n_name").alias("supp_nation"), col("l_orderkey"), col("l_extendedprice"), col("l_discount"),
                    col("l_shipdate"))

        return filteredNations.join(self.customer, col("n_nationkey") == col("c_nationkey")) \
            .join(self.orders, col("c_custkey") == col("o_custkey")) \
            .select(col("n_name").alias("cust_nation"), col("o_orderkey")) \
            .join(supplierNations, col("o_orderkey") == col("l_orderkey")) \
            .filter(((col("supp_nation") == "FRANCE") & (col("cust_nation") == "GERMANY"))
                    | ((col("supp_nation") == "GERMANY") & (col("cust_nation") == "FRANCE"))) \
            .select(col("supp_nation"), col("cust_nation"),
                    getYear(col("l_shipdate")).alias("l_year"),
                    decrease(col("l_extendedprice"), col("l_discount")).alias("volume")) \
            .groupBy(col("supp_nation"), col("cust_nation"), col("l_year")) \
            .agg(F.sum(col("volume")).alias("revenue")) \
            .sort(col("supp_nation"), col("cust_nation"), col("l_year"))

    def q8Common(self, decrease):
        getYear = udf(lambda x: x[0:4], StringType())
        isBrazil = udf(lambda x, y: (y if (x == "BRAZIL") else 0), FloatType())

        filteredRegions = self.region.filter(col("r_name") == "AMERICA")
        filteredOrders = self.orders.filter((col("o_orderdate") <= "1996-12-31") & (col("o_orderdate") >= "1995-01-01"))
        filteredParts = self.part.filter(col("p_type") == "ECONOMY ANODIZED STEEL")

        filteredNations = self.nation.join(self.supplier, col("n_nationkey") == col("s_nationkey"))

        filteredLineitems = self.lineitem.select(col("l_partkey"), col("l_suppkey"), col("l_orderkey"),
                                                 decrease(col("l_extendedprice"), col("l_discount")).alias("volume")) \
            .join(filteredParts, col("l_partkey") == col("p_partkey")) \
            .join(filteredNations, col("l_suppkey") == col("s_suppkey"))

        return self.nation.join(filteredRegions, col("n_regionkey") == col("r_regionkey")) \
            .select(col("n_nationkey")) \
            .join(self.customer, col("n_nationkey") == col("c_nationkey")) \
            .select(col("c_custkey")) \
            .join(filteredOrders, col("c_custkey") == col("o_custkey")) \
            .select(col("o_orderkey"), col("o_orderdate")) \
            .join(filteredLineitems, col("o_orderkey") == col("l_orderkey")) \
            .select(getYear(col("o_orderdate")).alias("o_year"), col("volume"),
                    isBrazil(col("n_name"), col("volume")).alias("case_volume")) \
            .groupBy(col("o_year")) \
            .agg((F.sum(col("case_volume")) / F.sum(col("volume"))).alias("mkt_share")) \
            .sort(col("o_year"))

    def q8(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())
        return self.q8Common(decrease)

    def q9(self):
        getYear = udf(lambda x: x[0:4], StringType())
        expression = udf(lambda x, y, v, w: x * (1 - y) - (v * w), FloatType())

        lineitemParts = self.part.filter(col("p_name").contains("green")) \
            .join(self.lineitem, col("p_partkey") == col("l_partkey"))

        nationPartSuppliers = self.nation.join(self.supplier, col("n_nationkey") == col("s_nationkey"))

        return lineitemParts.join(nationPartSuppliers, col("l_suppkey") == col("s_suppkey")) \
            .join(self.partsupp, (col("l_suppkey") == col("ps_suppkey"))
                  & (col("l_partkey") == col("ps_partkey"))) \
            .join(self.orders, col("l_orderkey") == col("o_orderkey")) \
            .select(col("n_name"), getYear(col("o_orderdate")).alias("o_year"),
                    expression(col("l_extendedprice"), col("l_discount"),
                               col("ps_supplycost"), col("l_quantity")).alias("amount")) \
            .groupBy(col("n_name"), col("o_year")) \
            .agg(F.sum(col("amount"))) \
            .sort(col("n_name"), col("o_year").desc())

    def q10(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        filteredLineitems = self.lineitem.filter(col("l_returnflag") == "R")

        return self.orders.filter((col("o_orderdate") < "1994-01-01") & (col("o_orderdate") >= "1993-10-01")) \
            .join(self.customer, col("o_custkey") == col("c_custkey")) \
            .join(self.nation, col("c_nationkey") == col("n_nationkey")) \
            .join(filteredLineitems, col("o_orderkey") == col("l_orderkey")) \
            .select(col("c_custkey"), col("c_name"),
                    decrease(col("l_extendedprice"), col("l_discount")).alias("volume"),
                    col("c_acctbal"), col("n_name"), col("c_address"), col("c_phone"), col("c_comment")) \
            .groupBy(col("c_custkey"), col("c_name"), col("c_acctbal"), col("c_phone"), col("n_name"), col("c_address"),
                     col("c_comment")) \
            .agg(F.sum(col("volume")).alias("revenue")) \
            .sort(col("revenue").desc()) \
            .limit(20)

    def q11(self):
        multiplication = udf(lambda x, y: x * y, FloatType())
        division = udf(lambda x: x * 0.0001, FloatType())

        nationPartSuppliers = self.nation.filter(col("n_name") == "GERMANY") \
            .join(self.supplier, col("n_nationkey") == col("s_nationkey")) \
            .select(col("s_suppkey")) \
            .join(self.partsupp, col("s_suppkey") == col("ps_suppkey")) \
            .select(col("ps_partkey"), multiplication(col("ps_supplycost"), col("ps_availqty")).alias("value"))

        aggregatedValue = nationPartSuppliers.agg(F.sum(col("value")).alias("total_value"))

        return nationPartSuppliers.groupBy(col("ps_partkey")).agg(F.sum(col("value")).alias("part_value")) \
            .join(aggregatedValue, col("part_value") > division(col("total_value"))) \
            .sort(col("part_value").desc())

    def q12(self):
        highPriority = udf(lambda x: (1 if ((x == "1-URGENT") or (x == "2-HIGH")) else 0), IntegerType())
        lowPriority = udf(lambda x: (1 if ((x != "1-URGENT") and (x != "2-HIGH")) else 0), IntegerType())

        return self.lineitem.filter(((col("l_shipmode") == "MAIL") | (col("l_shipmode") == "SHIP"))
                             & (col("l_commitdate") < col("l_receiptdate"))
                             & (col("l_shipdate") < col("l_commitdate"))
                             & (col("l_receiptdate") >= "1994-01-01")
                             & (col("l_receiptdate") < "1995-01-01")) \
            .join(self.orders, col("l_orderkey") == col("o_orderkey")) \
            .select(col("l_shipmode"), col("o_orderpriority")) \
            .groupBy(col("l_shipmode")) \
            .agg(F.sum(highPriority(col("o_orderpriority"))).alias("sum_highorderpriority"),
                 F.sum(lowPriority(col("o_orderpriority"))).alias("sum_loworderpriority")) \
            .sort(col("l_shipmode"))

    def q13(self):
        special_regex = re.compile(".*special.*requests.*")
        special = udf(lambda x: special_regex.match(x) is not None, BooleanType())

        return self.customer.join(self.orders, (col("c_custkey") == col("o_custkey"))
                           & ~special(col("o_comment")), "left_outer") \
            .groupBy(col("c_custkey")) \
            .agg(F.count(col("o_orderkey")).alias("c_count")) \
            .groupBy(col("c_count")) \
            .agg(F.count(col("*")).alias("custdist")) \
            .sort(col("custdist").desc(), col("c_count").desc())

    def q14(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())
        promotion = udf(lambda x, y: (y if (x.startswith("PROMO")) else 0), FloatType())

        return self.part.join(self.lineitem, (col("l_partkey") == col("p_partkey"))
                       & (col("l_shipdate") >= "1995-09-01")
                       & (col("l_shipdate") < "1995-10-01")) \
            .select(col("p_type"), decrease(col("l_extendedprice"), col("l_discount")).alias("value")) \
            .agg(F.sum(promotion(col("p_type"), col("value"))) * 100 / F.sum(col("value")))

    def q15(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        revenue = self.lineitem.filter((col("l_shipdate") >= "1996-01-01")
                                       & (col("l_shipdate") < "1996-04-01")) \
            .select(col("l_suppkey"), decrease(col("l_extendedprice"), col("l_discount")).alias("value")) \
            .groupBy(col("l_suppkey")) \
            .agg(F.sum(col("value")).alias("total"))

        return revenue.agg(F.max(col("total")).alias("max_total")) \
            .join(revenue, col("max_total") == col("total")) \
            .join(self.supplier, col("l_suppkey") == col("s_suppkey")) \
            .select(col("s_suppkey"), col("s_name"), col("s_address"), col("s_phone"), col("total")) \
            .sort(col("s_suppkey"))

    def q16(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())
        polished = udf(lambda x: x.startswith("MEDIUM POLISHED"), BooleanType())

        complains_regex = re.compile(".*Customer.*Complaints.*")
        complains = udf(lambda x: complains_regex.match(x) is not None, BooleanType())

        numbers_regex = re.compile("^(49|14|23|45|19|3|36|9)$")
        numbers = udf(lambda x: numbers_regex.match(str(x)) is not None, BooleanType())

        filteredParts = self.part.filter((col("p_brand") != "Brand#45")
                                         & (~polished(col("p_type")))
                                         & numbers(col("p_size"))) \
            .select(col("p_partkey"), col("p_brand"), col("p_type"), col("p_size"))

        return self.supplier.filter(~complains(col("s_comment"))) \
            .join(self.partsupp, col("s_suppkey") == col("ps_suppkey")) \
            .select(col("ps_partkey"), col("ps_suppkey")) \
            .join(filteredParts, col("ps_partkey") == col("p_partkey")) \
            .groupBy(col("p_brand"), col("p_type"), col("p_size")) \
            .agg(F.countDistinct(col("ps_suppkey")).alias("supplier_count")) \
            .sort(col("supplier_count").desc(), col("p_brand"), col("p_type"), col("p_size"))

    def q17(self):
        multiplier = udf(lambda x: x * 0.2)

        filteredLineitems = self.lineitem.select(col("l_partkey"), col("l_quantity"), col("l_extendedprice"))

        filteredParts = self.part.filter((col("p_brand") == "Brand#23") & (col("p_container") == "MED BOX")) \
            .select(col("p_partkey")) \
            .join(self.lineitem, col("p_partkey") == col("l_partkey"), "left_outer")

        return filteredParts.groupBy(col("p_partkey")) \
            .agg(multiplier(F.avg(col("l_quantity"))).alias("avg_quantity")) \
            .select(col("p_partkey").alias("key"), col("avg_quantity")) \
            .join(filteredParts, col("key") == col("p_partkey")) \
            .filter(col("l_quantity") < col("avg_quantity")) \
            .agg((F.sum(col("l_extendedprice")) / 7.0).alias("avg_yearly"))

    def q18(self):
        return self.lineitem.groupBy(col("l_orderkey")) \
            .agg(F.sum(col("l_quantity")).alias("sum_quantity")) \
            .filter(col("sum_quantity") > 300) \
            .select(col("l_orderkey").alias("key"), col("sum_quantity")) \
            .join(self.orders, col("o_orderkey") == col("key")) \
            .join(self.lineitem, col("o_orderkey") == col("l_orderkey")) \
            .join(self.customer, col("c_custkey") == col("o_custkey")) \
            .select(col("l_quantity"), col("c_name"), col("c_custkey"), col("o_orderkey"), col("o_orderdate"),
                    col("o_totalprice")) \
            .groupBy(col("c_name"), col("c_custkey"), col("o_orderkey"), col("o_orderdate"), col("o_totalprice")) \
            .agg(F.sum(col("l_quantity"))) \
            .sort(col("o_totalprice").desc(), col("o_orderdate")).limit(100)

    def q19(self):
        decrease = udf(lambda x, y: x * (1 - y), FloatType())

        sm_regex = re.compile("SM CASE|SM BOX|SM PACK|SM PKG")
        sm = udf(lambda x: sm_regex.match(x) is not None, BooleanType())

        med_regex = re.compile("MED BAG|MED BOX|MED PKG|MED PACK")
        med = udf(lambda x: med_regex.match(x) is not None, BooleanType())

        lg_regex = re.compile("LG CASE|LG BOX|LG PACK|LG PKG")
        lg = udf(lambda x: lg_regex.match(x) is not None, BooleanType())

        return self.part.join(self.lineitem, col("l_partkey") == col("p_partkey")) \
            .filter(((col("l_shipmode") == "AIR")
                     | (col("l_shipmode") == "AIR REG"))
                    & (col("l_shipinstruct") == "DELIVER IN PERSON")) \
            .filter(((col("p_brand") == "Brand#12")
                     & (sm(col("p_container")))
                     & (col("l_quantity") >= 1)
                     & (col("l_quantity") <= 11)
                     & (col("p_size") >= 1)
                     & (col("p_size") <= 5))
                    | ((col("p_brand") == "Brand#23")
                       & (med(col("p_container")))
                       & (col("l_quantity") >= 10)
                       & (col("l_quantity") <= 20)
                       & (col("p_size") >= 1)
                       & (col("p_size") <= 10))
                    | ((col("p_brand") == "Brand#34")
                       & (lg(col("p_container")))
                       & (col("l_quantity") >= 20)
                       & (col("l_quantity") <= 30)
                       & (col("p_size") >= 1)
                       & (col("p_size") <= 15))) \
            .select(decrease(col("l_extendedprice"), col("l_discount")).alias("volume")) \
            .agg(F.sum(col("volume")).alias("revenue"))

    def q20(self):
        forest = udf(lambda x: x.startswith("forest"), BooleanType())

        filteredLineitems = self.lineitem.filter(
            (col("l_shipdate") >= "1994-01-01") & (col("l_shipdate") < "1995-01-01")) \
            .groupBy(col("l_partkey"), col("l_suppkey")) \
            .agg((F.sum(col("l_quantity")) * 0.5).alias("sum_quantity"))

        filteredNations = self.nation.filter(col("n_name") == "CANADA")

        nationSuppliers = self.supplier.select(col("s_suppkey"), col("s_name"), col("s_nationkey"), col("s_address")) \
            .join(filteredNations, col("s_nationkey") == col("n_nationkey"))

        return self.part.filter(forest(col("p_name"))) \
            .select(col("p_partkey")).distinct() \
            .join(self.partsupp, col("p_partkey") == col("ps_partkey")) \
            .join(filteredLineitems, (col("ps_suppkey") == col("l_suppkey")) & (col("ps_partkey") == col("l_partkey"))) \
            .filter(col("ps_availqty") > col("sum_quantity")) \
            .select(col("ps_suppkey")).distinct() \
            .join(nationSuppliers, col("ps_suppkey") == col("s_suppkey")) \
            .select(col("s_name"), col("s_address")) \
            .sort(col("s_name"))

    def q21(self):
        filteredSuppliers = self.supplier.select(col("s_suppkey"), col("s_nationkey"), col("s_name"))

        selectedLineitems = self.lineitem.select(col("l_suppkey"), col("l_orderkey"), col("l_receiptdate"),
                                                 col("l_commitdate"))
        filteredLineitems = selectedLineitems.filter(col("l_receiptdate") > col("l_commitdate"))

        selectedGroupedLineItems = selectedLineitems.groupBy(col("l_orderkey")) \
            .agg(F.countDistinct(col("l_suppkey")).alias("suppkey_count"), F.max(col("l_suppkey")).alias("suppkey_max")) \
            .select(col("l_orderkey").alias("key"), col("suppkey_count"), col("suppkey_max"))

        filteredGroupedLineItems = filteredLineitems.groupBy(col("l_orderkey")) \
            .agg(F.countDistinct(col("l_suppkey")).alias("suppkey_count"), F.max(col("l_suppkey")).alias("suppkey_max")) \
            .select(col("l_orderkey").alias("key"), col("suppkey_count"), col("suppkey_max"))

        filteredOrders = self.orders.select(col("o_orderkey"), col("o_orderstatus")) \
            .filter(col("o_orderstatus") == "F")

        return self.nation.filter(col("n_name") == "SAUDI ARABIA") \
            .join(filteredSuppliers, col("n_nationkey") == col("s_nationkey")) \
            .join(filteredLineitems, col("s_suppkey") == col("l_suppkey")) \
            .join(filteredOrders, col("l_orderkey") == col("o_orderkey")) \
            .join(selectedGroupedLineItems, col("l_orderkey") == col("key")) \
            .filter(col("suppkey_count") > 1) \
            .select(col("s_name"), col("l_orderkey"), col("l_suppkey")) \
            .join(filteredGroupedLineItems, col("l_orderkey") == col("key"), "left_outer") \
            .select(col("s_name"), col("l_orderkey"), col("l_suppkey"), col("suppkey_count"), col("suppkey_max")) \
            .filter((col("suppkey_count") == 1)
                    & (col("l_suppkey") == col("suppkey_max"))) \
            .groupBy(col("s_name")) \
            .agg(F.count(col("l_suppkey")).alias("numwait")) \
            .sort(col("numwait").desc(), col("s_name")).limit(100)

    def q22(self):
        substring = udf(lambda x: x[0:2], StringType())

        phone_regex = re.compile("^(13|31|23|29|30|18|17)$")
        phone = udf(lambda x: phone_regex.match(x) is not None, BooleanType())

        filteredCustomers = self.customer.select(col("c_acctbal"), col("c_custkey"),
                                                 substring(col("c_phone")).alias("cntrycode")) \
            .filter(phone(col("cntrycode")))

        customerAverage = filteredCustomers.filter(col("c_acctbal") > 0.0) \
            .agg(F.avg(col("c_acctbal")).alias("avg_acctbal"))

        return self.orders.groupBy(col("o_custkey")) \
            .agg(col("o_custkey")) \
            .select(col("o_custkey")) \
            .join(filteredCustomers, col("o_custkey") == col("c_custkey"), "right_outer") \
            .filter(col("o_custkey").isNull()) \
            .join(customerAverage) \
            .filter(col("c_acctbal") > col("avg_acctbal")) \
            .groupBy(col("cntrycode")) \
            .agg(F.count(col("c_acctbal")).alias("numcust"), F.sum(col("c_acctbal")).alias("totalacctbal")) \
            .sort(col("cntrycode"))