// scalastyle:off
/*
 * This file is copied from:
 * https://github.com/ssavvides/tpch-spark/blob/master/src/main/scala/Q20.scala
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
import org.apache.spark.sql.functions.first
import org.apache.spark.sql.functions.sum
import org.apache.spark.sql.functions.udf

import com.intel.analytics.bigdl.ppml.PPMLContext

/**
 * TPC-H Query 20
 * Savvas Savvides <savvas@purdue.edu>
 *
 */
class Q20 extends TpchQuery {

  override def execute(sc: PPMLContext, schemaProvider: TpchSchemaProvider): DataFrame = {

    // this is used to implicitly convert an RDD to a DataFrame.
    val sqlContext = sc.getSparkSession.sqlContext
    import sqlContext.implicits._
    import schemaProvider._

    val forest = udf { (x: String) => x.startsWith("forest") }

    val flineitem = lineitem.filter($"l_shipdate" >= "1994-01-01" && $"l_shipdate" < "1995-01-01")
      .groupBy($"l_partkey", $"l_suppkey")
      .agg((sum($"l_quantity") * 0.5).as("sum_quantity"))

    val fnation = nation.filter($"n_name" === "CANADA")
    val nat_supp = supplier.select($"s_suppkey", $"s_name", $"s_nationkey", $"s_address")
      .join(fnation, $"s_nationkey" === fnation("n_nationkey"))

    part.filter(forest($"p_name"))
      .select($"p_partkey").distinct
      .join(partsupp, $"p_partkey" === partsupp("ps_partkey"))
      .join(flineitem, $"ps_suppkey" === flineitem("l_suppkey") &&
        $"ps_partkey" === flineitem("l_partkey"))
      .filter($"ps_availqty" > $"sum_quantity")
      .select($"ps_suppkey").distinct
      .join(nat_supp, $"ps_suppkey" === nat_supp("s_suppkey"))
      .select($"s_name", $"s_address")
      .sort($"s_name")
  }

}
