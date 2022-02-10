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
import org.apache.spark.sql.types.{StructField, StructType, LongType}
import org.apache.spark.sql.{SQLContext, SparkSession, Row}
import org.apache.spark.SparkContext

/*
    The dataset is tab separated with the following schema: <label> <integer feature 1> … <integer feature 13> <categorical feature 1> … <categorical feature 26>
    We set missing value to -999.
    Categorical feature is in hexadecimal format and we convert them into long type.
*/
class Task extends Serializable{

  val default_missing_value = "-999"

  def rowToLibsvm(row: Row): String = {
    0 until row.length flatMap {
      case 0 => Some(row(0).toString)
      case i if row(i) == null => Some(default_missing_value)
      case i => Some( (if (i < 14) row(i) else java.lang.Long.parseLong(row(i).toString, 16)).toString )
    } mkString " "
  }
}

object xgbClassifierTrainingExampleOnCriteoClickLogsDataset {

  val feature_nums = 39

  def main(args: Array[String]): Unit = {
    if (args.length < 5) {
      println("Usage: program input_path modelsave_path num_threads num_round max_depth")
      sys.exit(1)
    }

    val sc = NNContext.initNNContext()
    val spark = SQLContext.getOrCreate(sc)
    val task = new Task()

    val input_path = args(0) // path to data
    val modelsave_path = args(1) // save model to this path
    val num_threads = args(2).toInt // xgboost threads
    val num_round = args(3).toInt //  train round
    val max_depth = args(4).toInt // tree max depth

    // read csv files to dataframe
    var df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", "\t").csv(input_path)
    // preprocess data
    val processedRdd = df.rdd.map(task.rowToLibsvm)

    // declare schema
    var structFieldArray = new Array[StructField](feature_nums+1)
    for(i <- 0 to feature_nums) {
      structFieldArray(i) = StructField("_c" + i.toString, LongType, true)
    }
    var schema =  new StructType(structFieldArray)

    // convert RDD to RDD[Row]
    val rowRDD = processedRdd.map(_.split(" ")).map(row => Row.fromSeq(
      for{
        i <- 0 to feature_nums
      } yield {
        row(i).toLong
      }
    ))
    // RDD[Row] to Dataframe
    df = spark.createDataFrame(rowRDD,schema)

    val stringIndexer = new StringIndexer()
      .setInputCol("_c0")
      .setOutputCol("classIndex")
      .fit(df)
    val labelTransformed = stringIndexer.transform(df).drop("_c0")

    var inputCols = new Array[String](feature_nums)
    for(i <- 0 to feature_nums-1){
      inputCols(i) = "_c" + (i+1).toString
    }

    val vectorAssembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val xgbInput = vectorAssembler.transform(labelTransformed).select("features","classIndex")
    // randomly split dataset to (train, eval1, eval2, test) in proportion 6:2:1:1
    val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    // use scala tracker
    val xgbParam = Map("tracker_conf" -> TrackerConf(0L, "scala"),
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2),
    )

    val xgbClassifier = new XGBClassifier(xgbParam)
    xgbClassifier.setFeaturesCol("features")
    xgbClassifier.setLabelCol("classIndex")
    xgbClassifier.setNumClass(2)
    xgbClassifier.setNumWorkers(1)
    xgbClassifier.setMaxDepth(max_depth)
    xgbClassifier.setNthread(num_threads)
    xgbClassifier.setNumRound(num_round)
    xgbClassifier.setTreeMethod("auto")
    xgbClassifier.setObjective("multi:softprob")
    xgbClassifier.setTimeoutRequestWorkers(180000L)

    // start training model
    val xgbClassificationModel = xgbClassifier.fit(train)
    xgbClassificationModel.save(modelsave_path)

    sc.stop()
  }
}
