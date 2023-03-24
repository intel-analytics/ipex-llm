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

package com.intel.analytics.bigdl.dllib.example.nnframes.lightGBM

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nnframes.LightGBMClassifier
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier => MLightGBMClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import scopt.OptionParser

object LgbmClassifierTrain {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LGBMParams()
    Utils.parser.parse(args, defaultParams).foreach { params =>
      val sc = NNContext.initNNContext("LGBM example")
      val spark = SQLContext.getOrCreate(sc)

      val schema = new StructType(Array(
        StructField("sepal length", DoubleType, true),
        StructField("sepal width", DoubleType, true),
        StructField("petal length", DoubleType, true),
        StructField("petal width", DoubleType, true),
        StructField("class", StringType, true)))
      val df = spark.read.schema(schema).csv(params.inputPath).repartition(params.nPartition)

      val stringIndexer = new StringIndexer()
        .setInputCol("class")
        .setOutputCol("classIndex")
        .fit(df)
      val labelTransformed = stringIndexer.transform(df).drop("class")
      // compose all feature columns as vector
      val vectorAssembler = new VectorAssembler().
        setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
        setOutputCol("features")
      val dfinput = vectorAssembler.transform(labelTransformed).select("features",
        "classIndex")

      val Array(train, test) = dfinput.randomSplit(Array(0.8, 0.2))
      val classifier = new LightGBMClassifier()
      classifier.setFeaturesCol("features")
      classifier.setLabelCol("classIndex")
      classifier.setNumIterations(params.numIterations)
      classifier.setNumLeaves(params.numLeaves)
      classifier.setMaxDepth(params.maxDepth)
      classifier.setLambdaL1(params.lamda1)
      classifier.setLambdaL2(params.lamda2)
      classifier.setBaggingFreq(params.bagFreq)
      classifier.setMaxBin(params.maxBin)
      classifier.setNumIterations(params.numIterations)

      val model = classifier.fit(train)
      val predictions = model.transform(test)

      predictions.show(10)

      val evaluatorMulti = new MulticlassClassificationEvaluator()
        .setLabelCol("classIndex")
        .setMetricName("accuracy")

      val acc = evaluatorMulti.evaluate(predictions)
      println("acc:", acc)
      model.saveNativeModel(params.modelSavePath)

      sc.stop()
    }
  }
}

private object Utils {

  case class LGBMParams(
    inputPath: String = "iris/iris.data",
    numLeaves: Int = 10,
    maxDepth: Int = 6,
    lamda1: Double = 0.01,
    lamda2: Double = 0.01,
    bagFreq: Int = 5,
    minDataInLeaf: Int = 20,
    maxBin: Int = 255,
    numIterations: Int = 100,
    modelSavePath: String = "/tmp/lgbm/classifier",
    nPartition: Int = 4)

  val parser = new OptionParser[LGBMParams]("LGBM example") {
    opt[String]("inputPath")
      .text(s"inputPath")
      .action((x, c) => c.copy(inputPath = x))
    opt[Int]("numLeaves")
      .text(s"numLeaves")
      .action((x, c) => c.copy(numLeaves = x))
    opt[Int]("maxDepth")
      .text(s"maxDepth")
      .action((x, c) => c.copy(maxDepth = x))
    opt[Double]("lamda1")
      .text(s"lamda1")
      .action((x, c) => c.copy(lamda1 = x))
    opt[Double]("lamda2")
      .text(s"lamda2")
      .action((x, c) => c.copy(lamda2 = x))
    opt[Int]("bagFreq")
      .text(s"bagFreq")
      .action((x, c) => c.copy(bagFreq = x))
    opt[Int]("minDataInLeaf")
      .text(s"minDataInLeaf")
      .action((x, c) => c.copy(minDataInLeaf = x))
    opt[Int]("maxBin")
      .text(s"maxBin")
      .action((x, c) => c.copy(maxBin = x))
    opt[Int]('n', "numIterations")
      .text(s"numIterations")
      .action((x, c) => c.copy(numIterations = x))
    opt[String]("modelSavePath")
      .text(s"modelSavePath")
      .action((x, c) => c.copy(modelSavePath = x))
    opt[Int]('p', "partition")
      .text("The number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }
}
