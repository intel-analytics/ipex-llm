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
import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, AES_CBC_PKCS5PADDING, PLAIN_TEXT}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SparkSession}
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
    val filteredData = df.filter(col("BANK").isin(bankFilter:_*) &&
      (col("MONTH") >= to_date(lit(startDate), "MMM-yy")) &&
      (col("MONTH") <= last_day(to_date(lit(endDate), "MMM-yy"))))

    // Method 1: Total number of each category for each month
    val categoryData = filteredData.groupBy("MONTH")
      .agg(sum(when(col("RENT") > 0, col("RENT"))).as("RENT"),
        sum(when(col("FOOD") > 0, col("FOOD"))).as("FOOD"),
        sum(when(col("Transport") > 0, col("Transport"))).as("Transport"),
        sum(when(col("Clothing") > 0, col("Clothing"))).as("Clothing"),
        sum(when(col("Other") > 0, col("Other"))).as("Other"))
      .toJSON
      .write
      .mode("overwrite")
      .text(outputFilePath+"/output1.json")

    // Method 2: Total number of income and expense for each month
    val incomeExpenseData = filteredData.groupBy("MONTH")
      .agg(sum(when(col("INCOME") > 0, col("INCOME"))).as("INCOME"),
        sum(when(col("EXPENSE") > 0, col("EXPENSE"))).as("EXPENSE"))
      .toJSON
      .write
      .mode("overwrite")
      .text(outputFilePath+"/output1.json")

    // Method 3: Total number of RENT, FOOD, Transport, Clothing and Other
    val totalData = filteredData.agg(sum(when(col("RENT") > 0, col("RENT"))).as("RENT"),
      sum(when(col("FOOD") > 0, col("FOOD"))).as("FOOD"),
      sum(when(col("Transport") > 0, col("Transport"))).as("Transport"),
      sum(when(col("Clothing") > 0, col("Clothing"))).as("Clothing"),
      sum(when(col("Other") > 0, col("Other"))).as("Other"))
      .toJSON
      .write
      .mode("overwrite")
      .text(outputFilePath+"/output1.json")

    // Stop SparkSession
    sparkSession.stop()

  }
}

