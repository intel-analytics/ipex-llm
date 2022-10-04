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
import com.intel.analytics.bigdl.dllib.nnframes.XGBClassifier
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier => MLightGBMClassifier}
import com.intel.analytics.bigdl.dllib.nnframes.LightGBMClassifier
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{SQLContext, SparkSession}
import scopt.OptionParser

object LgbmClassifierTrain {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LGBMParams()
    Utils.parser.parse(args, defaultParams).foreach { params =>
      // val spark = SparkSession.builder().appName("LGBM example").getOrCreate()

      val sc = NNContext.initNNContext("LGBM example")
      val spark = SQLContext.getOrCreate(sc)
      val train_df = spark.read.parquet(params.inputPath + "/train").repartition(10)
      val test_df = spark.read.parquet(params.inputPath + "/test").repartition(10)

      train_df.show(10)

      val classifier = new LightGBMClassifier()
      classifier.setObjective("binary")
      classifier.setNumIterations(params.numIterations)
      val model = classifier.fit(train_df)
      val predictions = model.transform(test_df)

      val mclassifier = new MLightGBMClassifier()
      mclassifier.setObjective("binary")
      mclassifier.setNumIterations(params.numIterations)
      val mmodel = mclassifier.fit(train_df)
      val mpredictions = mmodel.transform(test_df)

      predictions.show(10)
      mpredictions.show(10)

      val evaluatorBin = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")

      val evaluatorMulti = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setMetricName("accuracy")

      val auc = evaluatorBin.evaluate(predictions)
      val acc = evaluatorMulti.evaluate(predictions)
      val mauc = evaluatorBin.evaluate(mpredictions)
      val macc = evaluatorMulti.evaluate(mpredictions)

      println("auc: " + auc)
      println("acc: " + acc)
      println("mauc: " + mauc)
      println("macc: " + macc)
    }
  }
}

private object Utils {

  case class LGBMParams(
    inputPath: String = "/Users/guoqiong/intelWork/data/tweet/xgb_processed",
    numIterations: Int = 100,
    modelSavePath: String = "/tmp/lgbm/classifier")

  val parser = new OptionParser[LGBMParams]("LGBM example") {
    opt[String]("inputPath")
      .text(s"inputPath")
      .action((x, c) => c.copy(inputPath = x))
    opt[Int]('n', "numIterations")
      .text(s"numIterations")
      .action((x, c) => c.copy(numIterations = x))
    opt[String]("modelSavePath")
      .text(s"modelSavePath")
      .action((x, c) => c.copy(modelSavePath = x))
  }
}
