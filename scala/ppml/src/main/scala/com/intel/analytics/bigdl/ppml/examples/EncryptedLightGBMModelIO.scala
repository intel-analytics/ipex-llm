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

import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.microsoft.azure.synapse.ml.lightgbm._
import com.intel.analytics.bigdl.ppml.crypto.AES_CBC_PKCS5PADDING
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.PLAIN_TEXT
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object EncryptedLightGBMModelIO extends Supportive {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val sc = PPMLContext.initPPMLContext(sparkSession)

    val Array(train, test) = timing("1/4 load data") {
      val schema = new StructType(Array(
        StructField("sepal length", DoubleType, true),
        StructField("sepal width", DoubleType, true),
        StructField("petal length", DoubleType, true),
        StructField("petal width", DoubleType, true),
        StructField("class", StringType, true)))

      val df = sc.read(PLAIN_TEXT).schema(schema).csv(args(0))
      val stringIndexer = new StringIndexer()
        .setInputCol("class")
        .setOutputCol("classIndex")
        .fit(df)
      val labelTransformed = stringIndexer.transform(df).drop("class")
      val vectorAssembler = new VectorAssembler().
        setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
        setOutputCol("features")
      val dfinput = vectorAssembler.transform(labelTransformed)
        .select("features", "classIndex")
      dfinput.randomSplit(Array(0.8, 0.2))
    }
    
    val model = timing("2/4 create a LightGBMClassifier and fit a LightGBMClassificationModel") {
      val classifier = new LightGBMClassifier()
      classifier.setFeaturesCol("features")
      classifier.setLabelCol("classIndex")
      classifier.setNumIterations(100)
      classifier.setNumLeaves(10)
      classifier.setMaxDepth(6)
      classifier.setLambdaL1(0.01)
      classifier.setLambdaL2(0.01)
      classifier.setBaggingFreq(5)
      classifier.setMaxBin(255)
      val lgbmClassificationModel = classifier.fit(train)
      lgbmClassificationModel
    }

    timing("3/4 save trained model in ciphtertext") {  
      sc.saveLightGBMModel(model = model,
          path = "./lgbmClassification.model", cryptoMode = AES_CBC_PKCS5PADDING)
    }

    timing("4/4 load the encrypted model and use it to predict") {
      val reloadedModel = sc.loadLightGBMClassificationModel(
            modelPath = "./lgbmClassification.model", cryptoMode = AES_CBC_PKCS5PADDING)
      val predictions = reloadedModel.transform(test)
      predictions.show(10)
    }
  } 
}
