// scalastyle:off
/*
 * This file is copied from:
 * https://github.com/ssavvides/tpch-spark/blob/master/src/main/scala/Q21.scala
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
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions.max
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions.udf

import com.intel.analytics.bigdl.ppml.PPMLContext

/**
 * TPC-H Query 21
 * Savvas Savvides <savvas@purdue.edu>
 *
 */
class Q21 extends TpchQuery {

  override def execute(sc: PPMLContext, schemaProvider: TpchSchemaProvider): DataFrame = {

    // this is used to implicitly convert an RDD to a DataFrame.
    val sqlContext = sc.getSparkSession.sqlContext
    import sqlContext.implicits._
    import schemaProvider._

    val fsupplier = supplier.select($"s_suppkey", $"s_nationkey", $"s_name")

    val plineitem = lineitem.select($"l_suppkey", $"l_orderkey", $"l_receiptdate", $"l_commitdate")
    // cache

    val flineitem = plineitem.filter($"l_receiptdate" > $"l_commitdate")
    // cache

    val line1 = plineitem.groupBy($"l_orderkey")
      .agg(countDistinct($"l_suppkey").as("suppkey_count"), max($"l_suppkey").as("suppkey_max"))
      .select($"l_orderkey".as("key"), $"suppkey_count", $"suppkey_max")

    val line2 = flineitem.groupBy($"l_orderkey")
      .agg(countDistinct($"l_suppkey").as("suppkey_count"), max($"l_suppkey").as("suppkey_max"))
      .select($"l_orderkey".as("key"), $"suppkey_count", $"suppkey_max")

    val forder = order.select($"o_orderkey", $"o_orderstatus")
      .filter($"o_orderstatus" === "F")

    nation.filter($"n_name" === "SAUDI ARABIA")
      .join(fsupplier, $"n_nationkey" === fsupplier("s_nationkey"))
      .join(flineitem, $"s_suppkey" === flineitem("l_suppkey"))
      .join(forder, $"l_orderkey" === forder("o_orderkey"))
      .join(line1, $"l_orderkey" === line1("key"))
      .filter($"suppkey_count" > 1 || ($"suppkey_count" == 1 && $"l_suppkey" == $"max_suppkey"))
      .select($"s_name", $"l_orderkey", $"l_suppkey")
      .join(line2, $"l_orderkey" === line2("key"), "left_outer")
      .select($"s_name", $"l_orderkey", $"l_suppkey", $"suppkey_count", $"suppkey_max")
      .filter($"suppkey_count" === 1 && $"l_suppkey" === $"suppkey_max")
      .groupBy($"s_name")
      .agg(count($"l_suppkey").as("numwait"))
      .sort($"numwait".desc, $"s_name")
      .limit(100)
  }

}
