/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.nnframes

import java.io.File

import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{LBFGS, Loss, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.MatToTensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.reflect.io.Path
import scala.util.Random

class NNClassifierSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    Random.setSeed(42)
    RNG.setSeed(42)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "NNClasifier" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("pascal/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val transformedDF = new NNImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")
      .transform(imageDF)
      .withColumn("label", lit(2.0f))
    val estimator = new NNClassifier(
      Inception_v1(1000), ClassNLLCriterion[Float](), Array(3, 224, 224))
      .setBatchSize(1)
      .setEndWhen(Trigger.maxIteration(1))
    estimator.fit(transformedDF)
  }

  "An NNClasifierModel" should "return same results after saving and loading" in {
    val data = sqlContext.createDataFrame(smallData).toDF("features", "label")
    val module = new Sequential[Double]().add(Linear[Double](6, 2)).add(LogSoftMax[Double])
    val nnModel = new NNClassifierModel[Double](module, Array(6))
    val result = nnModel.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted

    val tmpFile = File.createTempFile("DLModel", "bigdl")
    val filePath = File.createTempFile("DLModel", "bigdl").getPath + Random.nextLong().toString
    nnModel.setBatchSize(10).setFeatureSize(Array(10, 100))
      .setFeaturesCol("test123").setPredictionCol("predict123")
    nnModel.write.overwrite().save(filePath)
    val nnModel2 = try {
      NNClassifierModel.load(filePath)
    } finally {
     Path(tmpFile).deleteRecursively()
     Path(filePath).deleteRecursively()
    }
    nnModel2.uid shouldEqual nnModel.uid
    nnModel2.getBatchSize shouldEqual nnModel.getBatchSize
    nnModel2.getFeaturesCol shouldEqual nnModel.getFeaturesCol
    nnModel2.getPredictionCol shouldEqual nnModel.getPredictionCol
    nnModel2.getFeatureSize shouldEqual nnModel.getFeatureSize
    nnModel2.setFeatureSize(Array(6)).setFeaturesCol("features").setPredictionCol("prediction")
    val result2 = nnModel2.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted
    result2 shouldEqual result
  }

  "An NNClassifier" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val classifier = new NNClassifier[Float](model, criterion, Array(6))
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(13)
      .setFeaturesCol("abc")
      .setTrainSummary(new TrainSummary("/tmp", "1"))
      .setValidationSummary(new ValidationSummary("/tmp", "2"))
      .setValidation(Trigger.maxIteration(3), df, Array(new Loss[Float]()), 2)
    val copied = classifier.copy(ParamMap.empty)
    assert(classifier.model ne copied.model)
    assert(classifier.criterion ne copied.criterion)
    assert(classifier.featureSize ne copied.featureSize)

    assert(classifier.model == copied.model)
    assert(classifier.criterion == copied.criterion)
    assert(classifier.featureSize.deep == copied.featureSize.deep)
    NNEstimatorSpec.compareParams(classifier, copied)
    val estVal = classifier.getValidation.get
    val copiedVal = copied.getValidation.get
    assert(estVal._1 == copiedVal._1)
    assert(estVal._2 == copiedVal._2)
    assert(estVal._3.deep == copiedVal._3.deep)
    assert(estVal._4 == copiedVal._4)

    // train Summary and validation Summary are not copied since they are not thread-safe and cannot
    // be shared among estimators
    assert(copied.getTrainSummary.isEmpty)
    assert(copied.getValidationSummary.isEmpty)
  }

  "A NNClassifierModel" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("abc", "la")
    val classifier = new NNClassifier[Float](model, criterion, Array(6))
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(3)
      .setFeaturesCol("abc")
      .setLabelCol("la")

    val nnModel = classifier.fit(df)
    val copied = nnModel.copy(ParamMap.empty)
    assert(copied.isInstanceOf[NNClassifierModel[Float]])
    assert(nnModel.model ne copied.model)
    assert(nnModel.featureSize ne copied.featureSize)

    assert(nnModel.model == copied.model)
    assert(nnModel.featureSize.deep == copied.featureSize.deep)
    NNEstimatorSpec.compareParams(nnModel, copied)
  }
}
