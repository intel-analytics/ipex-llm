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

package com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost

import ml.dmlc.xgboost4j.scala.spark.TrackerConf
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nnframes.XGBClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType, LongType}
import org.apache.spark.sql.{SQLContext, SparkSession, Row}
import org.apache.spark.SparkContext

object xgbClassifierTrainingExampleOnCriteoClickLogsDataset {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println("Usage: program input_path modelsave_path num_threads")
      sys.exit(1)
    }

    val sc = NNContext.initNNContext()
    val spark = SQLContext.getOrCreate(sc)
    val task = new Task()

    val input_path = args(0) // path to data
    val modelsave_path = args(1) // save model to this path
    val num_threads = args(2).toInt // xgboost threads

    // var df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", " ").csv(input_path)
    var df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", "\t").csv(input_path)
    val processedRdd = df.rdd.map(task.rowToLibsvm)

    var structFieldArray = new Array[StructField](40)
    for(i <- 0 to 39){
      structFieldArray(i) = StructField("_c" + i.toString, if(i<14) IntegerType else LongType, true)
    //   structFieldArray(i) = StructField("_c" + i.toString, LongType, true)
    }
    var schema =  new StructType(structFieldArray)
  
    // val decryptionRDD = decryption.flatMap(_.split("\n"))

    val rowRDD = processedRdd.map(_.split(" ")).map(row => Row.fromSeq(
      for{
        i <- 0 to 39
      } yield {
        if(i<14) row(i).toInt else row(i).toLong
        // row(i).toLong
      }
    ))

    df = spark.createDataFrame(rowRDD,schema)

    val stringIndexer = new StringIndexer()
      .setInputCol("_c0")
      .setOutputCol("classIndex")
      .fit(df)
    val labelTransformed = stringIndexer.transform(df).drop("_c0")

    var inputCols = new Array[String](39)
    for(i <- 0 to 38){
      inputCols(i) = "_c" + (i+1).toString
    }

    val vectorAssembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val xgbInput = vectorAssembler.transform(labelTransformed).select("features","classIndex")
    val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    val xgbParam = Map("tracker_conf" -> TrackerConf(0L, "scala"),
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2),
    )

    val xgbClassifier = new XGBClassifier(xgbParam)
    xgbClassifier.setFeaturesCol("features")
    xgbClassifier.setLabelCol("classIndex")
    xgbClassifier.setNumClass(2)
    xgbClassifier.setNumWorkers(1)
    xgbClassifier.setMaxDepth(2)
    xgbClassifier.setNthread(num_threads)
    xgbClassifier.setNumRound(10)
    xgbClassifier.setTreeMethod("auto")
    xgbClassifier.setObjective("multi:softprob")
    xgbClassifier.setTimeoutRequestWorkers(180000L)
    val xgbClassificationModel = xgbClassifier.fit(train)
    xgbClassificationModel.save(modelsave_path)

    sc.stop()
  }
}
