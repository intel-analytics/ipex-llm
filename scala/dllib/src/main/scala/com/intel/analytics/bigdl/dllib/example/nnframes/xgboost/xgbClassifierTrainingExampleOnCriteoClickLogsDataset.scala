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

package com.intel.analytics.bigdl.dllib.example.nnframes.xgboost

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nnframes.XGBClassifier
import ml.dmlc.xgboost4j.scala.spark.TrackerConf
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import scopt.OptionParser
import org.slf4j.{Logger, LoggerFactory}

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
                   modelSavePath: String = "/host/data/model",
                   numThreads: Int = 2,
                   numRound: Int = 100,
                   maxDepth: Int = 2,
                   numWorkers: Int = 1
                 )

object xgbClassifierTrainingExampleOnCriteoClickLogsDataset {

  val feature_nums = 39

  def main(args: Array[String]): Unit = {
    val log: Logger = LoggerFactory.getLogger(this.getClass)


    // parse params and set value

    val params = parser.parse(args, new Params).get
    val trainingDataPath = params.trainingDataPath // path to data
    val modelSavePath = params.modelSavePath // save model to this path
    val numThreads = params.numThreads // xgboost threads
    val numRound = params.numRound //  train round
    val maxDepth = params.maxDepth // tree max depth
    val numWorkers = params.numWorkers //  Workers num

    val sc = NNContext.initNNContext()
    val spark = SQLContext.getOrCreate(sc)
    val task = new Task()

    val tStart = System.nanoTime()
    // read csv files to dataframe
    var df = spark.read.option("header", "false").
      option("inferSchema", "true").option("delimiter", "\t").csv(trainingDataPath)

    val tBeforePreprocess = System.nanoTime()
    var elapsed = (tBeforePreprocess - tStart) / 1000000000.0f // second
    log.info("--reading data time is " + elapsed + " s")
    // preprocess data
    val processedRdd = df.rdd.map(task.rowToLibsvm)

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
    df = spark.createDataFrame(rowRDD, schema)

    val stringIndexer = new StringIndexer()
      .setInputCol("_c0")
      .setOutputCol("classIndex")
      .fit(df)
    val labelTransformed = stringIndexer.transform(df).drop("_c0")

    var inputCols = new Array[String](feature_nums)
    for (i <- 0 to feature_nums - 1) {
      inputCols(i) = "_c" + (i + 1).toString
    }

    val vectorAssembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val xgbInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")
    // randomly split dataset to (train, eval1, eval2, test) in proportion 6:2:1:1
    val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    train.cache().count()
    eval1.cache().count()
    eval2.cache().count()

    val tBeforeTraining = System.nanoTime()
    elapsed = (tBeforeTraining - tBeforePreprocess) / 1000000000.0f // second
    log.info("--preprocess time is " + elapsed + " s")
    // use scala tracker
    val xgbParam = Map("tracker_conf" -> TrackerConf(0L, "scala"),
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2)
    )

    val xgbClassifier = new XGBClassifier(xgbParam)
    xgbClassifier.setFeaturesCol("features")
    xgbClassifier.setLabelCol("classIndex")
    xgbClassifier.setNumClass(2)
    xgbClassifier.setNumWorkers(numWorkers)
    xgbClassifier.setMaxDepth(maxDepth)
    xgbClassifier.setNthread(numThreads)
    xgbClassifier.setNumRound(numRound)
    xgbClassifier.setTreeMethod("auto")
    xgbClassifier.setObjective("multi:softprob")
    xgbClassifier.setTimeoutRequestWorkers(180000L)

    // start training model
    val xgbClassificationModel = xgbClassifier.fit(train)

    val tAfterTraining = System.nanoTime()
    elapsed = (tAfterTraining - tBeforeTraining) / 1000000000.0f // second
    log.info("--training time is " + elapsed + " s")

    xgbClassificationModel.save(modelSavePath)

    val tAfterSave = System.nanoTime()
    elapsed = (tAfterSave - tAfterTraining) / 1000000000.0f // second
    log.info("--model save time is " + elapsed + " s")
    elapsed = (tAfterSave - tStart) / 1000000000.0f // second
    log.info("--end-to-end time is " + elapsed + " s")
    sc.stop()
  }

  val parser: OptionParser[Params] = new OptionParser[Params]("input xgboost config") {
    opt[String]('i', "trainingDataPath")
      .text("trainingData Path")
      .action((v, p) => p.copy(trainingDataPath = v))
      .required()

    opt[String]('s', "modelSavePath")
      .text("savePath of model")
      .action((v, p) => p.copy(modelSavePath = v))
      .required()

    opt[Int]('t', "numThreads")
      .text("threads num")
      .action((v, p) => p.copy(numThreads = v))

    opt[Int]('r', "numRound")
      .text("Round num")
      .action((v, p) => p.copy(numRound = v))

    opt[Int]('d', "maxDepth")
      .text("maxDepth")
      .action((v, p) => p.copy(maxDepth = v))

    opt[Int]('w', "numWorkers")
      .text("Workers num")
      .action((v, p) => p.copy(numWorkers = v))

  }
}
