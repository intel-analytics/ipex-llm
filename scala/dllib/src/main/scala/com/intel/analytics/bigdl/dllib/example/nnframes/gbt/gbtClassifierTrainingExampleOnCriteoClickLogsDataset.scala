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

package com.intel.analytics.bigdl.dllib.example.nnframes.gbt

import com.intel.analytics.bigdl.dllib.NNContext
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser


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
                   maxIter: Int = 100,
                   maxDepth: Int = 2
                 )

object gbtClassifierTrainingExampleOnCriteoClickLogsDataset {

  val feature_nums = 39

  def main(args: Array[String]): Unit = {

    val log: Logger = LoggerFactory.getLogger(this.getClass)


    // parse params and set value

    val params = parser.parse(args, new Params).get
    val trainingDataPath = params.trainingDataPath // path to data
    val modelSavePath = params.modelSavePath // save model to this path
    val maxIter = params.maxIter //  train max Iter
    val maxDepth = params.maxDepth // tree max depth

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

    val gbtInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")
    // randomly split dataset to (train, eval1, eval2, test) in proportion 6:2:1:1
    val Array(train, eval1, eval2, test) = gbtInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    train.cache().count()
    eval1.cache().count()
    eval2.cache().count()

    val tBeforeTraining = System.nanoTime()
    elapsed = (tBeforeTraining - tBeforePreprocess) / 1000000000.0f // second
    log.info("--preprocess time is " + elapsed + " s")
    // use scala tracker
    //    val gbtParam = Map("tracker_conf" -> TrackerConf(0L, "scala"),
    //      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2)
    //    )

    // Train a GBT model.
    val gbtClassifier = new GBTClassifier()
    gbtClassifier.setFeaturesCol("features")
    gbtClassifier.setLabelCol("classIndex")
    gbtClassifier.setMaxDepth(maxDepth)
    gbtClassifier.setMaxIter(maxIter)
    gbtClassifier.setFeatureSubsetStrategy("auto")

    // Train model. This also runs the indexer.
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
    sc.stop()
  }

  val parser: OptionParser[Params] = new OptionParser[Params]("input gbt config") {
    opt[String]('i', "trainingDataPath")
      .text("trainingData Path")
      .action((v, p) => p.copy(trainingDataPath = v))
      .required()

    opt[String]('s', "modelSavePath")
      .text("savePath of model")
      .action((v, p) => p.copy(modelSavePath = v))
      .required()

    opt[Int]('I', "maxIter")
      .text("maxIter")
      .action((v, p) => p.copy(maxIter = v))

    opt[Int]('d', "maxDepth")
      .text("maxDepth")
      .action((v, p) => p.copy(maxDepth = v))

  }
}

