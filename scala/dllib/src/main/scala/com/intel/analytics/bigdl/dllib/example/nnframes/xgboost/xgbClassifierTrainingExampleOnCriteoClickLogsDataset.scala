package com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost

import ml.dmlc.xgboost4j.scala.spark.TrackerConf
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nnframes.XGBClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.SparkContext

object xgbClassifierTrainingExampleOnCriteoClickLogsDataset {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println("Usage: program input_path modelsave_path num_threads")
      sys.exit(1)
    }

    val sc = NNContext.initNNContext()
    val spark = SQLContext.getOrCreate(sc)

    val input_path = args(0) // path to data
    val modelsave_path = args(1) // save model to this path
    val num_threads = args(2).toInt // xgboost threads

    var df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", " ").csv(input_path)

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

    spark.stop()
    sc.stop()
  }
}
