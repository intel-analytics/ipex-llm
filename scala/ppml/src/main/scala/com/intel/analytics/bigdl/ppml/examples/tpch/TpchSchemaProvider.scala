// scalastyle:off
/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is adapted from
 * https://github.com/ssavvides/tpch-spark/blob/master/src/main/scala/TpchSchemaProvider.scala
 *
 * MIT License
 *
 * Copyright (c) 2015 Savvas Savvides, ssavvides@us.ibm.com, savvas@purdue.edu
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

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode

// TPC-H table schemas
case class Customer(
                     c_custkey: Long,
                     c_name: String,
                     c_address: String,
                     c_nationkey: Long,
                     c_phone: String,
                     c_acctbal: Double,
                     c_mktsegment: String,
                     c_comment: String)

case class Lineitem(
                     l_orderkey: Long,
                     l_partkey: Long,
                     l_suppkey: Long,
                     l_linenumber: Long,
                     l_quantity: Double,
                     l_extendedprice: Double,
                     l_discount: Double,
                     l_tax: Double,
                     l_returnflag: String,
                     l_linestatus: String,
                     l_shipdate: String,
                     l_commitdate: String,
                     l_receiptdate: String,
                     l_shipinstruct: String,
                     l_shipmode: String,
                     l_comment: String)

case class Nation(
                   n_nationkey: Long,
                   n_name: String,
                   n_regionkey: Long,
                   n_comment: String)

case class Order(
                  o_orderkey: Long,
                  o_custkey: Long,
                  o_orderstatus: String,
                  o_totalprice: Double,
                  o_orderdate: String,
                  o_orderpriority: String,
                  o_clerk: String,
                  o_shippriority: Long,
                  o_comment: String)

case class Part(
                 p_partkey: Long,
                 p_name: String,
                 p_mfgr: String,
                 p_brand: String,
                 p_type: String,
                 p_size: Long,
                 p_container: String,
                 p_retailprice: Double,
                 p_comment: String)

case class Partsupp(
                     ps_partkey: Long,
                     ps_suppkey: Long,
                     ps_availqty: Long,
                     ps_supplycost: Double,
                     ps_comment: String)

case class Region(
                   r_regionkey: Long,
                   r_name: String,
                   r_comment: String)

case class Supplier(
                     s_suppkey: Long,
                     s_name: String,
                     s_address: String,
                     s_nationkey: Long,
                     s_phone: String,
                     s_acctbal: Double,
                     s_comment: String)

class TpchSchemaProvider(sc: PPMLContext, inputDir: String, cryptoMode: CryptoMode) {

  // this is used to implicitly convert an RDD to a DataFrame.
  val sqlContext = sc.getSparkSession().sqlContext

  import sqlContext.implicits._

  val dfMap = Map(
        "customer" -> sc.textFile(inputDir + "/customer.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Customer(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim,
            p(5).trim.toDouble, p(6).trim, p(7).trim))
          .toDF(),

        "lineitem" -> sc.textFile(inputDir + "/lineitem.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Lineitem(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong,
            p(4).trim.toDouble, p(5).trim.toDouble, p(6).trim.toDouble, p(7).trim.toDouble,
            p(8).trim, p(9).trim, p(10).trim, p(11).trim, p(12).trim, p(13).trim,
            p(14).trim, p(15).trim))
          .toDF(),

        "nation" -> sc.textFile(inputDir + "/nation.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Nation(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim))
          .toDF(),

        "region" -> sc.textFile(inputDir + "/region.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Region(p(0).trim.toLong, p(1).trim, p(2).trim))
          .toDF(),

        "order" -> sc.textFile(inputDir + "/orders.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Order(p(0).trim.toLong, p(1).trim.toLong, p(2).trim, p(3).trim.toDouble,
            p(4).trim, p(5).trim, p(6).trim, p(7).trim.toLong, p(8).trim))
          .toDF(),

        "part" -> sc.textFile(inputDir + "/part.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Part(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim,
            p(5).trim.toLong, p(6).trim, p(7).trim.toDouble, p(8).trim))
          .toDF(),

        "partsupp" -> sc.textFile(inputDir + "/partsupp.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Partsupp(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong,
            p(3).trim.toDouble, p(4).trim))
          .toDF(),

        "supplier" -> sc.textFile(inputDir + "/supplier.tbl*", cryptoMode = cryptoMode)
          .map(_.split('|'))
          .map(p =>
          Supplier(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong,
            p(4).trim, p(5).trim.toDouble, p(6).trim))
          .toDF()
  )

  // for implicits
  val customer = dfMap.get("customer").get
  val lineitem = dfMap.get("lineitem").get
  val nation = dfMap.get("nation").get
  val region = dfMap.get("region").get
  val order = dfMap.get("order").get
  val part = dfMap.get("part").get
  val partsupp = dfMap.get("partsupp").get
  val supplier = dfMap.get("supplier").get

  dfMap.foreach {
    case (key, value) => value.createOrReplaceTempView(key)
  }
}
