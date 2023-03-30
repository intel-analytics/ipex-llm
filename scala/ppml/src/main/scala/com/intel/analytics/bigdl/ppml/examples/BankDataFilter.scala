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
import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object BankDataFilter extends Supportive {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()

    val sc = PPMLContext.initPPMLContext(sparkSession)

    // Get user inputs
    val csvFilePath = args(0)
    val bankFilter = args(1).split(",")
    val startDate = args(2)
    val endDate = args(3)
    val outputFilePath = args(4)


    // Read CSV file
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true")
      .csv(csvFilePath)

    // Filter data based on user inputs
    val filteredData = df.filter(col("BANK").isin(bankFilter: _*)
        && date_format(col("MONTH"), "yyyy-MM-dd").between(startDate, endDate))

    filteredData.show()

    // Table 1: Total number of each category for each month
    filteredData.groupBy("MONTH")
      .agg(sum(col("Rent")).as("Rent"),
        sum(col("Food")).as("Food"),
        sum(col("Transport")).as("Transport"),
        sum(col("Clothing")).as("Clothing"),
        sum(col("Other")).as("Other"))
      .coalesce(1)
      .write
      .mode("overwrite")
      .json(outputFilePath + "/categoryDate")

    // Table 2: Total number of income and expense for each month
    filteredData.groupBy("MONTH")
      .agg(sum(col("INCOME")).as("INCOME"),
        sum(col("EXPENSE")).as("EXPENSE"))
      .coalesce(1)
      .write
      .mode("overwrite")
      .json(outputFilePath + "/incomeExpenseDate")

    // Table 3: Total number of RENT, FOOD, Transport, Clothing and Other
    filteredData.agg(sum(col("Rent")).as("Rent"),
      sum(col("Food")).as("Food"),
      sum(col("Transport")).as("Transport"),
      sum(col("Clothing")).as("Clothing"),
      sum(col("Other")).as("Other"))
      .coalesce(1)
      .write
      .mode("overwrite")
      .json(outputFilePath + "/totalCategory")

    // Stop SparkSession
    sparkSession.stop()

  }
}

