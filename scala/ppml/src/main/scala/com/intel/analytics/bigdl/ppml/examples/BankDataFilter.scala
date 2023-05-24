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

package com.intel.analytics.bigdl.ppml.examples

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.PLAIN_TEXT
import com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample.getClass
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, Supportive}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory

object BankDataFilter extends Supportive {
  def main(args: Array[String]): Unit = {
    val logger = LoggerFactory.getLogger(getClass)
    val arguments = EncryptIOArguments.parser.parse(args, EncryptIOArguments()) match {
        case Some(arguments) =>
          logger.info(s"starting with $arguments"); arguments
        case None =>
          EncryptIOArguments.parser.failure("miss args, please see the usage info"); null
      }

    val sc = PPMLContext.initPPMLContext("BankDataFilter", arguments.ppmlArgs())

    val extraList = arguments.extras
    val bankFilter = extraList.head.split(",")
    val startDate = extraList.apply(1)
    val endDate = extraList.apply(2)

    logger.info(s"Loading Encrypted Data")

    // Read CSV file
    val df = sc.read(cryptoMode = arguments.inputEncryptMode,
      primaryKeyName = "defaultKey")
      .option("header", "true")
      .csv(arguments.inputPath)

    logger.info(s"Distributed Processing in SGX Enclaves")

    // Filter data based on user inputs
    val filteredData = df.filter(col("BANK").isin(bankFilter: _*)
        && date_format(col("MONTH"), "yyyy-MM-dd").between(startDate, endDate))

    // Table 1: Total number of each category for each month
    val categoryDateDf = filteredData.groupBy("MONTH")
      .agg(sum(col("Rent")).as("Rent"),
        sum(col("Food")).as("Food"),
        sum(col("Transport")).as("Transport"),
        sum(col("Clothing")).as("Clothing"),
        sum(col("Other")).as("Other"))
      .coalesce(1)
      sc.write(categoryDateDf,
        cryptoMode = arguments.outputEncryptMode,
        primaryKeyName = "defaultKey")
      .mode("overwrite")
      .json(arguments.outputPath + "/categoryDate")

    // Table 2: Total number of income and expense for each month
    val incomeExpenseDateDf = filteredData.groupBy("MONTH")
      .agg(sum(col("INCOME")).as("INCOME"),
        sum(col("EXPENSE")).as("EXPENSE"))
      .coalesce(1)
    sc.write(incomeExpenseDateDf,
      cryptoMode = arguments.outputEncryptMode,
      primaryKeyName = "defaultKey")
      .mode("overwrite")
      .json(arguments.outputPath + "/incomeExpenseDate")

    // Table 3: Total number of RENT, FOOD, Transport, Clothing and Other
    val totalCategoryDf = filteredData.agg(sum(col("Rent")).as("Rent"),
      sum(col("Food")).as("Food"),
      sum(col("Transport")).as("Transport"),
      sum(col("Clothing")).as("Clothing"),
      sum(col("Other")).as("Other"))
      .coalesce(1)
    sc.write(totalCategoryDf,
      cryptoMode = arguments.outputEncryptMode,
      primaryKeyName = "defaultKey")
      .mode("overwrite")
      .json(arguments.outputPath + "/totalCategory")

    logger.info(s"Saving Encrypted Result")
  }
}

