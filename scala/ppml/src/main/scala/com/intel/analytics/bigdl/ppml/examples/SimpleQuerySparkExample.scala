package com.intel.analytics.bigdl.ppml.examples

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import com.intel.analytics.bigdl.ppml.utils.EncryptIOArguments
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object SimpleQuerySparkExample {

  def main(args: Array[String]): Unit = {
    val logger = LoggerFactory.getLogger(getClass)

    // parse parameter
    val arguments = EncryptIOArguments.parser.parse(args, EncryptIOArguments()) match {
        case Some(arguments) => logger.info(s"starting with $arguments"); arguments
        case None => EncryptIOArguments.parser.failure("miss args, please see the usage info"); null
      }

    val sc = PPMLContext.initPPMLContext("SimpleQuery", arguments.ppmlArgs())

    // read kms args from spark-defaults.conf
    // val sc = PPMLContext.initPPMLContext("SimpleQuery")

    // load csv file to data frame with ppmlcontext.
    val df = sc.read(mode = arguments.inputEncryptMode).csv(arguments.inputPath + "/people.csv")

    // Select only the "name" column
    df.select("name").count()

    // Select everybody, but increment the age by 1
    df.select(df("name"), df("age") + 1).show()

    // Select Developer and records count
    val developers = df.filter(df("job") === "Developer" and df("age").between(20, 40)).toDF()
    developers.count()

    Map[String, DataFrame]({
      "developers" -> developers
    })

    // save data frame using spark kms context
    sc.write(developers, mode = arguments.outputEncryptMode).mode("overwrite").option("header", true).csv(arguments.outputPath)
  }
}
