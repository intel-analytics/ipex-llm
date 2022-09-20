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
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.kms.KMS_CONVENTION
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import java.io.File

/*
    The dataset is tab separated with the following schema:
    <label> <integer feature 1> … <integer feature 13>
      <categorical feature 1> … <categorical feature 26>
    We set missing value to -999.
    Categorical feature is in hexadecimal format and we convert them into long type.
*/
class Task extends Serializable {

  val default_missing_value = "-999"

  def rowToLibsvm(row: Row): String = {
    0 until row.length flatMap {
      case 0 => Some(row(0).toString)
      case i if row(i) == null => Some(default_missing_value)
      case i => Some((if (i < 14) row(i)
      else java.lang.Long.parseLong(row(i).toString, 16)).toString)
    } mkString " "
  }
}

case class Params(
                   trainingDataPath: String = "/host/data",
                   inputEncryptMode: CryptoMode = PLAIN_TEXT,
                   primaryKeyPath: String = "./primaryKeyPath",
                   dataKeyPath: String = "./dataKeyPath",
                   kmsType: String = KMS_CONVENTION.MODE_SIMPLE_KMS,
                   kmsServerIP: String = "0.0.0.0",
                   kmsServerPort: String = "5984",
                   ehsmAPPID: String = "ehsmAPPID",
                   ehsmAPIKEY: String = "ehsmAPIKEY",
                   simpleAPPID: String = "simpleAPPID",
                   simpleAPIKEY: String = "simpleAPIKEY",
                   modelSavePath: String = "/host/data/model",
                   maxIter: Int = 100,
                   maxDepth: Int = 2
                 ) {
  def ppmlArgs(): Map[String, String] = {
    val kmsArgs = scala.collection.mutable.Map[String, String]()
    kmsArgs("spark.bigdl.kms.type") = kmsType
    kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        kmsArgs("spark.bigdl.kms.ehs.ip") = kmsServerIP
        kmsArgs("spark.bigdl.kms.ehs.port") = kmsServerPort
        kmsArgs("spark.bigdl.kms.ehs.id") = ehsmAPPID
        kmsArgs("spark.bigdl.kms.ehs.key") = ehsmAPIKEY
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        kmsArgs("spark.bigdl.kms.simple.id") = simpleAPPID
        kmsArgs("spark.bigdl.kms.simple.key") = simpleAPIKEY
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    if (new File(primaryKeyPath).exists()) {
      kmsArgs("spark.bigdl.kms.key.primary") = primaryKeyPath
    }
    if (new File(dataKeyPath).exists()) {
      kmsArgs("spark.bigdl.kms.key.data") = dataKeyPath
    }
    kmsArgs.toMap
  }
}

object gbtClassifierTrainingExampleOnCriteoClickLogsDataset {

  val feature_nums = 39

  def main(args: Array[String]): Unit = {
    val log: Logger = LoggerFactory.getLogger(this.getClass)


    // parse params and set value

    val params = parser.parse(args, new Params).get
    val ppmlArgs = params.ppmlArgs() // args to create a PPMLContext
    val inputEncryptMode = params.inputEncryptMode
    val trainingDataPath = params.trainingDataPath // path to data
    val modelSavePath = params.modelSavePath // save model to this path
    val maxIter = params.maxIter //  train round
    val maxDepth = params.maxDepth // tree max depth

    val sc = PPMLContext.initPPMLContext("gbtClassifierTrainingExample", ppmlArgs)
    val spark = sc.getSparkSession()
    val task = new Task()

    val tStart = System.nanoTime()
    // read csv files to dataframe
    val csvDF = sc.read(inputEncryptMode)
      .option("header", "false")
      .option("inferSchema", "true")
      .option("delimiter", "\t")
      .csv(trainingDataPath)

    val tBeforePreprocess = System.nanoTime()
    var elapsed = (tBeforePreprocess - tStart) / 1000000000.0f // second
    log.info("--reading data time is " + elapsed + " s")
    // preprocess data
    val processedRdd = csvDF.rdd.map(task.rowToLibsvm)

    // declare schema
    var structFieldArray = new Array[StructField](feature_nums + 1)
    for (i <- 0 to feature_nums) {
      structFieldArray(i) = StructField("_c" + i.toString, LongType, true)
    }
    var schema = new StructType(structFieldArray)

    // convert RDD to RDD[Row]
    val rowRDD = processedRdd.map(_.split(" ")).map(row => Row.fromSeq(
      for {
        i <- 0 to feature_nums
      } yield {
        row(i).toLong
      }
    ))
    // RDD[Row] to Dataframe
    val df = spark.createDataFrame(rowRDD, schema)

    val stringIndexer = new StringIndexer()
      .setInputCol("_c0")
      .setOutputCol("classIndex")
      .fit(df)
    val labelTransformed = stringIndexer.transform(df).drop("_c0")

    val inputCols = new Array[String](feature_nums)
    for (i <- 0 to feature_nums - 1) {
      inputCols(i) = "_c" + (i + 1).toString
    }

    val vectorAssembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val gbtInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")
    // randomly split dataset to (train, eval1, eval2, test) in proportion 6:2:1:1
    val Array(train, eval1, eval2, test) = gbtInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    train.cache().count()
    eval1.cache().count()
    eval2.cache().count()

    val tBeforeTraining = System.nanoTime()
    elapsed = (tBeforeTraining - tBeforePreprocess) / 1000000000.0f // second
    log.info("--preprocess time is " + elapsed + " s")

    val gbtClassifier = new GBTClassifier()
    gbtClassifier.setFeaturesCol("features")
    gbtClassifier.setLabelCol("classIndex")
    gbtClassifier.setMaxDepth(maxDepth)
    gbtClassifier.setMaxIter(maxIter)
    gbtClassifier.setFeatureSubsetStrategy("auto")


    // start training model
    val gbtClassificationModel = gbtClassifier.fit(train)

    val tAfterTraining = System.nanoTime()
    elapsed = (tAfterTraining - tBeforeTraining) / 1000000000.0f // second
    log.info("--training time is " + elapsed + " s")

    gbtClassificationModel.write.overwrite().save(modelSavePath)

    val tAfterSave = System.nanoTime()
    elapsed = (tAfterSave - tAfterTraining) / 1000000000.0f // second
    log.info("--model save time is " + elapsed + " s")
    elapsed = (tAfterSave - tStart) / 1000000000.0f // second
    log.info("--end-to-end time is " + elapsed + " s")
  }

  val parser: OptionParser[Params] = new OptionParser[Params]("input GBT config") {
    opt[String]('i', "trainingDataPath")
      .text("trainingData Path")
      .action((v, p) => p.copy(trainingDataPath = v))
      .required()

    opt[String]('a', "inputEncryptMode")
      .action((v, p) => p.copy(inputEncryptMode = CryptoMode.parse(v)))
      .text("inputEncryptModeValue: plain_text/aes_cbc_pkcs5padding")
      .required()

    opt[String]('s', "modelSavePath")
      .text("savePath of model")
      .action((v, p) => p.copy(modelSavePath = v))
      .required()

    opt[String]('p', "primaryKeyPath")
      .action((v, p) => p.copy(primaryKeyPath = v))
      .text("primaryKeyPath")

    opt[String]('d', "dataKeyPath")
      .action((v, p) => p.copy(dataKeyPath = v))
      .text("dataKeyPath")

    opt[String]('k', "kmsType")
      .action((v, p) => p.copy(kmsType = v))
      .text("kmsType")

    opt[String]('g', "kmsServerIP")
      .action((v, p) => p.copy(kmsServerIP = v))
      .text("kmsServerIP")

    opt[String]('h', "kmsServerPort")
      .action((v, p) => p.copy(kmsServerPort = v))
      .text("kmsServerPort")

    opt[String]('j', "ehsmAPPID")
      .action((v, p) => p.copy(ehsmAPPID = v))
      .text("ehsmAPPID")

    opt[String]('k', "ehsmAPIKEY")
      .action((v, p) => p.copy(ehsmAPIKEY = v))
      .text("ehsmAPIKEY")

    opt[String]('s', "simpleAPPID")
      .action((v, p) => p.copy(simpleAPPID = v))
      .text("simpleAPPID")

    opt[String]('k', "simpleAPIKEY")
      .action((v, p) => p.copy(simpleAPIKEY = v))
      .text("simpleAPIKEY")

    opt[Int]('I', "maxIter")
      .text("maxIter")
      .action((v, p) => p.copy(maxIter = v))

    opt[Int]('d', "maxDepth")
      .text("maxDepth")
      .action((v, p) => p.copy(maxDepth = v))

  }
}

