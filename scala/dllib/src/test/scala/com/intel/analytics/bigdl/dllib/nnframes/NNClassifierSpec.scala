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
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, LBFGS, Loss, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.feature.image._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.io.Path
import scala.util.Random

class NNClassifierSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    val conf = Engine.createSparkConf().setAppName("Test NNClassifier").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    val seed = System.currentTimeMillis()
    Random.setSeed(seed)
    RNG.setSeed(seed)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "NNClassifier" should "has correct default params" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNClassifier(model, criterion, Array(10))
    assert(estimator.getFeaturesCol == "features")
    assert(estimator.getLabelCol == "label")
    assert(estimator.getMaxEpoch == 50)
    assert(estimator.getBatchSize == 1)
    assert(estimator.getLearningRate == 1e-3)
    assert(estimator.getLearningRateDecay == 0)
  }

  "NNClassifier" should "apply with differnt params" in {
    val model = Linear[Float](6, 2)
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    Seq(
      NNClassifier(model, criterion),
      NNClassifier(model, criterion, Array(6)),
      NNClassifier(model, criterion, SeqToTensor(Array(6)))
    ).foreach(c => c.setMaxEpoch(1).fit(df))
  }

  "NNClassifier" should "get reasonable accuracy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(6))
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setBatchSize(nRecords)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = classifier.fit(df)
    nnModel.isInstanceOf[NNClassifierModel[_]] should be(true)
    assert(nnModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
  }

  "NNClassifier" should "support model with Sigmoid" in {
    val model = new Sequential().add(Linear[Float](6, 10)).add(Linear[Float](10, 1))
      .add(Sigmoid[Float])
    val criterion = BCECriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(6))
      .setOptimMethod(new Adam[Float]())
      .setLearningRate(0.01)
      .setBatchSize(10)
      .setMaxEpoch(10)
    val data = sc.parallelize(smallData.map(t => (t._1, t._2 - 1.0)))
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = classifier.fit(df)
    nnModel.isInstanceOf[NNClassifierModel[_]] should be(true)
    val correctCount = nnModel.transform(df).where("prediction=label").count()
    assert(correctCount > nRecords * 0.8)
  }

  "NNClassifier" should "apply with size support different FEATURE types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(6))
      .setLearningRate(0.1)
      .setBatchSize(2)
      .setEndWhen(Trigger.maxIteration(2))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.map(_.toFloat), p._2))))
        .toDF("features", "label") // Array[Float]
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val nnModel = classifier.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "NNClassifier" should "support scalar FEATURE" in {
    val model = new Sequential().add(Linear[Float](1, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(1))
      .setLearningRate(0.1)
      .setBatchSize(2)
      .setEndWhen(Trigger.maxIteration(2))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head.toFloat, p._2))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val nnModel = classifier.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "NNClassifier" should "fit with adam and LBFGS" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    Seq(new LBFGS[Float], new Adam[Float]).foreach { optimMethod =>
      val classifier = NNClassifier(model, criterion, Array(6))
        .setBatchSize(nRecords)
        .setMaxEpoch(2)
        .setOptimMethod(optimMethod)
        .setLearningRate(0.1)
      val data = sc.parallelize(smallData)
      val df = sqlContext.createDataFrame(data).toDF("features", "label")
      val nnModel = classifier.fit(df)
      nnModel.isInstanceOf[NNClassifierModel[_]] should be(true)
    }
  }

  "NNClassifier" should "supports validation data and summary" in {
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val logdir = com.google.common.io.Files.createTempDir()
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(6))
      .setBatchSize(nRecords)
      .setEndWhen(Trigger.maxIteration(5))
      .setOptimMethod(new Adam[Float])
      .setLearningRate(0.1)
      .setValidation(Trigger.severalIteration(1), df, Array(new Loss[Float]()), 2)
      .setValidationSummary(ValidationSummary(logdir.getPath, "NNEstimatorValidation"))

    classifier.fit(df)
    val validationSummary = classifier.getValidationSummary.get
    val losses = validationSummary.readScalar("Loss")
    validationSummary.close()
    logdir.deleteOnExit()
  }

  "NNClassifier" should "get the same classification result with BigDL model" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val model = LeNet5(10)

    // init
    val valTrans = NNClassifierModel(model, Array(28, 28))
      .setBatchSize(4)

    val tensorBuffer = new ArrayBuffer[Data]()
    val input = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())
    val target = model.forward(input).toTensor[Float]

    // test against NNClassifierModel
    val inputArr = input.storage().array()
    val targetArr = target.max(2)._2.squeeze().storage().array()
    (0 until 10).foreach(i =>
      tensorBuffer.append(
        Data(targetArr(i), inputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))
    val rowRDD = sc.parallelize(tensorBuffer)
    val testData = sqlContext.createDataFrame(rowRDD)
    assert(valTrans.transform(testData).where("prediction=label").count() == testData.count())
    tensorBuffer.clear()
  }

  "NNClassifier" should "works in ML pipeline" in {
    val appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        .setMax(1).setMin(-1)
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = NNClassifier(model, criterion)
        .setBatchSize(nRecords)
        .setOptimMethod(new LBFGS[Float]())
        .setLearningRate(0.1)
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")
      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      assert(pipelineModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
    }
  }

  "NNClasifier" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("pascal/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc)
      .withColumn("label", lit(2.0f))
    assert(imageDF.count() == 1)
    val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104, 1, 1, 1) -> ImageMatToTensor() -> ImageFeatureToTensor()

    val estimator = NNClassifier(Inception_v1(1000), ClassNLLCriterion[Float](), transformer)
      .setBatchSize(1)
      .setEndWhen(Trigger.maxIteration(1))
      .setFeaturesCol("image")
    estimator.fit(imageDF)
  }

  "NNClasifierModel" should "has default batchperthread as 4" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    Seq(new LBFGS[Float], new Adam[Float]).foreach { optimMethod =>
      val classifier = NNClassifier(model, criterion, Array(6))
        .setBatchSize(nRecords)
        .setMaxEpoch(2)
        .setOptimMethod(optimMethod)
        .setLearningRate(0.1)
      val data = sc.parallelize(smallData)
      val df = sqlContext.createDataFrame(data).toDF("features", "label")
      val nnModel = classifier.fit(df)
      nnModel.isInstanceOf[NNClassifierModel[_]] should be(true)
      nnModel.getBatchSize should be(4)
    }
  }

  "NNClasifierModel" should "return same results after saving and loading" in {
    val data = sqlContext.createDataFrame(smallData).toDF("features", "label")
    val module = new Sequential[Double]().add(Linear[Double](6, 2)).add(LogSoftMax[Double])
    val nnModel = NNClassifierModel(module)
    val result = nnModel.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted

    val tmpFile = File.createTempFile("NNModel", "zoo")
    val filePath = File.createTempFile("NNModel", "zoo").getPath + Random.nextLong().toString
    nnModel.setBatchSize(10).setFeaturesCol("test123").setPredictionCol("predict123")
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
    nnModel2.setFeaturesCol("features").setPredictionCol("prediction")
    val result2 = nnModel2.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted
    result2 shouldEqual result
  }

  "NNClassifierModel" should "apply with differnt params" in {
    val model = Linear[Float](6, 2)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    Seq(
      NNClassifierModel(model),
      NNClassifierModel(model, Array(6)),
      NNClassifierModel(model, SeqToTensor(Array(6)))
    ).foreach { e =>
      e.transform(df).count()
      assert(e.getBatchSize == 4)
    }
  }

  "NNClassifier" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val classifier = NNClassifier(model, criterion)
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

    assert(classifier.model == copied.model)
    assert(classifier.criterion == copied.criterion)
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

  "NNClassifierModel" should "construct with sampleTransformer" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val sampleTransformer = SeqToTensor(Array(6)) -> TensorToSample()

    val nnModel = NNClassifierModel(model).setBatchSize(nRecords)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    assert(nnModel.transform(df).count() == nRecords)
  }

  "NNClassifierModel" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("abc", "la")
    val classifier = NNClassifier(model, criterion)
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(3)
      .setFeaturesCol("abc")
      .setLabelCol("la")

    val nnModel = classifier.fit(df)
    val copied = nnModel.copy(ParamMap.empty)
    assert(copied.isInstanceOf[NNClassifierModel[_]])
    assert(nnModel.model ne copied.model)

    assert(nnModel.model == copied.model)
    NNEstimatorSpec.compareParams(nnModel, copied)
  }

  "NNClassifierModel" should "supports set Preprocessing" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val classifier = NNClassifier(model, criterion)
      .setBatchSize(31)
      .setMaxEpoch(1)

    val nnModel = classifier.fit(df)
    val newPreprocessing = ArrayToTensor(Array(6)) -> TensorToSample()
    nnModel.setSamplePreprocessing(newPreprocessing)
    assert(df.count() == nnModel.transform(df).count())
  }
}

private case class Data(label: Double, features: Array[Double])
