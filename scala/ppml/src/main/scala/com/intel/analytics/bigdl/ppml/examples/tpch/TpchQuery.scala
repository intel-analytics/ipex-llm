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
 * https://github.com/ssavvides/tpch-spark/blob/master/src/main/scala/TpchQuery.scala
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

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import org.apache.spark.sql._
import scala.collection.mutable.ListBuffer
import java.net.URI
import org.apache.hadoop.fs.{FileSystem, Path}
import scala.collection.mutable._

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode

/**
 * Parent class for TPC-H queries.
 *
 * Defines schemas for tables and reads pipe ("|") separated text files into these tables.
 *
 * Savvas Savvides <savvas@purdue.edu>
 *
 */
abstract class TpchQuery {

  // get the name of the class excluding dollar signs and package
  private def escapeClassName(className: String): String = {
    className.split("\\.").last.replaceAll("\\$", "")
  }

  def getName(): String = escapeClassName(this.getClass.getName)

  /**
   *  implemented in children classes and hold the actual query
   */
  def execute(sc: PPMLContext, tpchSchemaProvider: TpchSchemaProvider): DataFrame
}

object TpchQuery {

  def outputDF(df: DataFrame,
               outputDir: String,
               className: String,
               sc: PPMLContext,
               cryptoMode: CryptoMode): Unit = {

    if (outputDir == null || outputDir == "") {
      df.collect().foreach(println)
    }
    else {
      sc.write(df, cryptoMode)
        .mode("overwrite")
        .option("header", "true")
        .csv(outputDir + "/" + className)
    }
  }

  def executeQueries(sc: PPMLContext,
                     schemaProvider: TpchSchemaProvider,
                     queryNum: Int,
                     outputDir: String,
                     outputCryptoMode: CryptoMode): ListBuffer[(String, Float)] = {

    // if set write results to hdfs, if null write to stdout
    // val OUTPUT_DIR: String = "hdfs://172.168.10.103/tmp/output"
    // val OUTPUT_DIR: String = "file://" + new File(".").getAbsolutePath() + "/dbgen/output"

    val results = new ListBuffer[(String, Float)]

    var fromNum = 1;
    var toNum = 22;
    val queries = Queue[TpchQuery]()

    if (queryNum != 0) {
      var query = queryNum match {
        case 1 => new Q01()
        case 2 => new Q02()
        case 3 => new Q03()
        case 4 => new Q04()
        case 5 => new Q05()
        case 6 => new Q06()
        case 7 => new Q07()
        case 8 => new Q08()
        case 9 => new Q09()
        case 10 => new Q10()
        case 11 => new Q11()
        case 12 => new Q12()
        case 13 => new Q13()
        case 14 => new Q14()
        case 15 => new Q15()
        case 16 => new Q16()
        case 17 => new Q17()
        case 18 => new Q18()
        case 19 => new Q19()
        case 20 => new Q20()
        case 21 => new Q21()
        case 22 => new Q22()
      }
      queries.enqueue(query)
    } else {
      queries.enqueue(new Q01())
      queries.enqueue(new Q02())
      queries.enqueue(new Q03())
      queries.enqueue(new Q04())
      queries.enqueue(new Q05())
      queries.enqueue(new Q06())
      queries.enqueue(new Q07())
      queries.enqueue(new Q08())
      queries.enqueue(new Q09())
      queries.enqueue(new Q10())
      queries.enqueue(new Q11())
      queries.enqueue(new Q12())
      queries.enqueue(new Q13())
      queries.enqueue(new Q14())
      queries.enqueue(new Q15())
      queries.enqueue(new Q16())
      queries.enqueue(new Q17())
      queries.enqueue(new Q18())
      queries.enqueue(new Q19())
      queries.enqueue(new Q20())
      queries.enqueue(new Q21())
      queries.enqueue(new Q22())
    }

    while(queries.nonEmpty) {
      val query = queries.dequeue

      val t0 = System.nanoTime()

      outputDF(query.execute(sc, schemaProvider), outputDir, query.getName(), sc, outputCryptoMode)

      val t1 = System.nanoTime()

      val elapsed = (t1 - t0) / 1000000000.0f // second

      results += new Tuple2(query.getName(), elapsed)
    }

    return results
  }

  def main(args: Array[String]): Unit = {

    val inputDir = args(0)
    val outputDir = args(1)
    val inputCryptoMode = CryptoMode.parse(args(2))
    val outputCryptoMode = CryptoMode.parse(args(3))
    val queryNums = if (args.length > 4) {
      args.slice(4, args.length).map(_.toInt)
    } else {
      (1 to 22).toArray
    }
    println(s"Will run ${queryNums.mkString(" ")}")

    val conf = new SparkConf()
    val sc = PPMLContext.initPPMLContext(conf, "Simple Application")

    // read files from local FS
    // val INPUT_DIR = "file://" + new File(".").getAbsolutePath() + "/dbgen"

    // read from hdfs
    // val INPUT_DIR: String = "/dbgen"

    val schemaProvider = new TpchSchemaProvider(sc, inputDir, inputCryptoMode)

    val output = new ListBuffer[(String, Float)]
    for (queryNum <- queryNums) {
      output ++= executeQueries(sc, schemaProvider, queryNum, outputDir, outputCryptoMode)
      println(s"----------------$queryNum finished--------------------")
    }

    val hadoopConfig = sc.getSparkSession().sparkContext.hadoopConfiguration
    val fs: FileSystem = FileSystem.get(new URI(outputDir), hadoopConfig)
    val outputStream = fs.create(new Path(outputDir, "TIMES.txt"))
    output.foreach {
      case (key, value) => outputStream.writeBytes(f"${key}%s\t${value}%1.8f\n")
    }

    outputStream.close()
  }
}
