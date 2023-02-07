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

import java.util.Properties
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, Supportive}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession, SaveMode}
import org.slf4j.LoggerFactory

object MySQLSparkExample extends App {

    val dbUrl = this.args(0)

    val sc = PPMLContext.initPPMLContext("spark-mysql-io")

    val spark: SparkSession = SparkSession.builder
        .appName("spark-mysql-io")
        .getOrCreate

    val props = new Properties
    props.put("user", "root")
    props.put("password", "password")

    val mysqlURL = s"jdbc:mysql://$dbUrl?rewriteBatchedStatements=true"

    var inDF = spark.read.jdbc(mysqlURL, "people_in", props).coalesce(10)
    println(s"The people_in table contains ${inDF.count} records.")

    val outDF = inDF.drop("age")
                    .filter(inDF("job") === "Engineer")
                    .withColumnRenamed("job", "occupation")

    outDF.write.mode(SaveMode.Overwrite).jdbc(mysqlURL, "people_out", props)

    println(s"The people_out table contains ${outDF.count} records.")
    println("done!")

}
