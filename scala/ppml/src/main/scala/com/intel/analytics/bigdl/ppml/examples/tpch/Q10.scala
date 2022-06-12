// scalastyle:off
/*
 * This file is copied from:
 * https://github.com/ssavvides/tpch-spark/blob/master/src/main/scala/Q10.scala
 *
 * Copyright (c) 2015 Savvas Savvides, ssavvides@us.ibm.com, savvas@purdue.edu
 *
 * Licensed under the The MIT License:
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
// scalastyle:on

package com.intel.analytics.bigdl.ppml.examples.tpch

import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.sum
import org.apache.spark.sql.functions.udf

import com.intel.analytics.bigdl.ppml.PPMLContext

/**
 * TPC-H Query 10
 * Savvas Savvides <savvas@purdue.edu>
 *
 */
class Q10 extends TpchQuery {

  override def execute(sc: PPMLContext, schemaProvider: TpchSchemaProvider): DataFrame = {

    // this is used to implicitly convert an RDD to a DataFrame.
    val sqlContext = sc.getSparkSession.sqlContext
    import sqlContext.implicits._
    import schemaProvider._

    val decrease = udf { (x: Double, y: Double) => x * (1 - y) }

    val flineitem = lineitem.filter($"l_returnflag" === "R")

    order.filter($"o_orderdate" < "1994-01-01" && $"o_orderdate" >= "1993-10-01")
      .join(customer, $"o_custkey" === customer("c_custkey"))
      .join(nation, $"c_nationkey" === nation("n_nationkey"))
      .join(flineitem, $"o_orderkey" === flineitem("l_orderkey"))
      .select($"c_custkey", $"c_name",
        decrease($"l_extendedprice", $"l_discount").as("volume"),
        $"c_acctbal", $"n_name", $"c_address", $"c_phone", $"c_comment")
      .groupBy($"c_custkey", $"c_name", $"c_acctbal", $"c_phone", $"n_name",
        $"c_address", $"c_comment")
      .agg(sum($"volume").as("revenue"))
      .sort($"revenue".desc)
      .limit(20)
  }

}
