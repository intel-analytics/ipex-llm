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
 */

package com.intel.analytics.bigdl.serving.example

import com.intel.analytics.bigdl.serving.operator.ClusterServingFunction
import org.apache.flink.table.api.{EnvironmentSettings, TableEnvironment}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object ClusterServingFlinkSqlExample {
  case class ExampleParams(modelPath: String = null, inputFile: String = null)

  val parser = new OptionParser[ExampleParams]("Cluster Serving Operator Usage Example") {
    opt[String]('m', "modelPath")
      .text("Model Path of Cluster Serving")
      .action((x, params) => params.copy(modelPath = x))
      .required()
    opt[String]('i', "inputFile")
      .text("Input CSV of example")
      .action((x, params) => params.copy(inputFile = x))
      .required()
  }
  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {
    val arg = parser.parse(args, ExampleParams()).head
    val settings = EnvironmentSettings.newInstance().build()
    val tableEnv = TableEnvironment.create(settings)
    tableEnv.getConfig.addJobParameter("modelPath", arg.modelPath)
    tableEnv.createTemporarySystemFunction("ClusterServingFunction",
      new ClusterServingFunction())
    tableEnv.executeSql("CREATE TABLE Input (`uri` STRING, data STRING) WITH (" +
      s"'connector' = 'filesystem', 'path' = '${arg.inputFile}', 'format' = 'csv')")
    // run a SQL query on the Table and retrieve the result as a new Table
    val result = tableEnv.sqlQuery("SELECT * FROM Input")
    result.printSchema()
    tableEnv.executeSql("CREATE TABLE Print (data STRING) WITH ('connector'='print')")
    tableEnv.executeSql("INSERT INTO Print SELECT ClusterServingFunction(uri, data) FROM Input")
      .getJobClient.get().getJobExecutionResult(Thread.currentThread().getContextClassLoader).get()
  }
}
