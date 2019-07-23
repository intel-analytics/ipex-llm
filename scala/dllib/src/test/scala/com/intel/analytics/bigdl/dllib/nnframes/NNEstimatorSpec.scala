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

import com.google.common.io.Files
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.common.{TensorToSample, _}
import com.intel.analytics.zoo.feature.image._
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.reflect.io.Path
import scala.util.Random

class NNEstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    Random.setSeed(42)
    RNG.setSeed(42)
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator").setMaster("local[2]")
    sc = NNContext.initNNContext(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An NNEstimator" should "has correct default params" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(10), Array(1))
    assert(estimator.getFeaturesCol == "features")
    assert(estimator.getLabelCol == "label")
    assert(estimator.getMaxEpoch == 50)
    assert(estimator.getBatchSize == 1)
    assert(estimator.getLearningRate == 1e-3)
    assert(estimator.getLearningRateDecay == 0)
    assert(estimator.isCachingSample)
  }

  "NNEstimator" should "apply with differnt params" in {
    val model = Linear[Float](6, 2)
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    Seq(
      NNEstimator(model, criterion),
      NNEstimator(model, criterion, Array(6), Array(1)),
      NNEstimator(model, criterion, SeqToTensor(Array(6)), ScalarToTensor())
    ).foreach(e => e.setEndWhen(Trigger.maxIteration(1)).setBatchSize(2).fit(df))
  }

  "An NNEstimator" should "get reasonable accuracy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)
    nnModel.isInstanceOf[NNModel[Float]] should be(true)
    val predictionDF = nnModel.transform(df)
    predictionDF.show()
    predictionDF.printSchema()
    val correct = nnModel.transform(df).select("label", "prediction").rdd.filter {
      case Row(label: Double, prediction: Seq[_]) =>
        label == prediction.indexOf(prediction.asInstanceOf[Seq[Float]].max) + 1
    }.count()
    assert(correct > nRecords * 0.8)
  }

  "An NNEstimator" should "construct with sampleTransformer" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val sampleTransformer = FeatureLabelPreprocessing(SeqToTensor(Array(6)), ScalarToTensor())

    val estimator = new NNEstimator(model, criterion)
      .setSamplePreprocessing(sampleTransformer)
      .setBatchSize(nRecords)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setMaxEpoch(1)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)
    nnModel.isInstanceOf[NNModel[Float]] should be(true)
    val predictionDF = nnModel.transform(df)
    predictionDF.show()
    predictionDF.printSchema()
    assert(nnModel.transform(df).count() == nRecords)
  }

  "NNEstimator" should "apply with size support different data types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(2)
      // intentionally set low since this only validates data format compatibility
      .setEndWhen(Trigger.maxIteration(1))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.toSeq, p._2))))
        .toDF("features", "label"), // Seq[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.map(_.toFloat), p._2))))
        .toDF("features", "label"), // Array[Float]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.map(_.toInt), p._2))))
        .toDF("features", "label"), // Array[Int]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (Vectors.dense(p._1), p._2))))
        .toDF("features", "label") // MLlib Vector
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val nnModel = estimator.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support scalar FEATURE types" in {
    val model = new Sequential().add(Linear[Float](1, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(1), Array(1))
      .setBatchSize(2)
      // intentionally set low since this only validates data format compatibility
      .setEndWhen(Trigger.maxIteration(1))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head.toFloat, p._2))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val nnModel = estimator.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support different LABEL types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = MultiLabelSoftMarginCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(2))
      // intentionally set low since this only validates data format compatibitliy
      .setEndWhen(Trigger.maxIteration(1))
      .setBatchSize(2)

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, Array(p._2, p._2)))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1,
        Array(p._2.toFloat, p._2.toFloat))))).toDF("features", "label") // Array[Float]
    ).foreach { df =>
      val nnModel = estimator.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support scalar LABEL types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      // intentionally set low since this only validates data format compatibitliy
      .setEndWhen(Trigger.maxIteration(1))
      .setBatchSize(2)

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2.toFloat))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val nnModel = estimator.fit(df)
      nnModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "work with tensor data" in {

    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(10), Array(1))
      .setMaxEpoch(1)
      .setBatchSize(20)

    val featureData = Array.tabulate(100)(_ => Tensor(10))
    val labelData = Array.tabulate(100)(_ => Tensor(1).fill(1.0f))
    val miniBatch = sc.parallelize(
      featureData.zip(labelData).map(v =>
        MinibatchData(v._1.storage.array, v._2.storage.array))
    )
    val trainingDF: DataFrame = sqlContext.createDataFrame(miniBatch).toDF("features", "label")

    val nnModel = estimator.fit(trainingDF)
    nnModel.transform(trainingDF).collect()
  }

  "An NNEstimator" should "support different batchSize" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(52)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)
    nnModel.isInstanceOf[NNModel[_]] should be(true)
    assert(nnModel.transform(df).count() == nRecords)
  }

  "An NNModel" should "has default batchperthread as 4" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(52)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)
    nnModel.isInstanceOf[NNModel[_]] should be(true)
    assert(nnModel.getBatchSize == 4 * 2)
  }

  "An NNModel" should "support transform with different batchSize" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(20)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val nnModel = estimator.fit(df)
    assert(df.count() == nnModel.setBatchSize(51).transform(df).count())
  }

  "An NNModel" should "supports set Preprocessing" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(20)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val nnModel = estimator.fit(df)
    val newPreprocessing = ArrayToTensor(Array(6)) -> TensorToSample()
    nnModel.setSamplePreprocessing(newPreprocessing)
    assert(df.count() == nnModel.transform(df).count())
  }

  "An NNEstimator" should "throw exception with incorrect inputs" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val inputs = Array[String]("Feature data", "Label data")
    val estimator = NNEstimator(model, criterion, Array(10), Array(2, 1)).
      setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)
    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(), labelData.storage().array())
    ))
    var df: DataFrame = sqlContext.createDataFrame(miniBatch).toDF(inputs: _*)
    // Spark 1.6 and 2.0 throws different exception here
    intercept[Exception] {
      estimator.fit(df)
    }
  }

  "An NNEstimator" should "supports training summary" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(50)
      .setMaxEpoch(2)
      .setTrainSummary(TrainSummary(logdir.getPath, "DLEstimatorTrain"))
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)
    val trainSummary = estimator.getTrainSummary.get
    val losses = trainSummary.readScalar("Loss")
    assert(losses.length > 0)
    trainSummary.close()
    logdir.deleteOnExit()
  }

  "An NNEstimator" should "supports validation data and summary" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(4)
      .setEndWhen(Trigger.maxIteration(5))
      .setValidation(Trigger.severalIteration(1), df, Array(new Loss[Float]()), 2)
      .setValidationSummary(ValidationSummary(logdir.getPath, "NNEstimatorValidation"))

    val nnModel = estimator.fit(df)
    val validationSummary = estimator.getValidationSummary.get
    val losses = validationSummary.readScalar("Loss")
    assert(losses.length == 5)
    validationSummary.close()
    logdir.deleteOnExit()
  }

  "An NNEstimator" should "throw exception when EndWhen and MaxEpoch are set" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()

    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = NNEstimator( model, criterion, Array(6), Array(1))
      .setBatchSize(4)
      .setEndWhen(Trigger.maxIteration(5))
      .setMaxEpoch(5)

    intercept[Exception] {
      estimator.fit(df)
    }
  }

  "An NNEstimator" should "works in ML pipeline" in {
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        .setMax(1).setMin(-1)
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = NNEstimator( model, criterion, Array(6), Array(1))
        .setOptimMethod(new LBFGS[Float]())
        .setLearningRate(0.1)
        .setBatchSize(nRecords)
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")
      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      val correct = pipelineModel.transform(df).select("label", "prediction").rdd.filter {
        case Row(label: Double, prediction: Seq[_]) =>
          label == prediction.indexOf(prediction.asInstanceOf[Seq[Float]].max) + 1
      }.count()
      assert(correct > nRecords * 0.8)
    }
  }

  "NNModel" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("pascal/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()
    val featurizer = NNModel(Inception_v1(1000), transformer)
      .setBatchSize(1)
      .setFeaturesCol("image")
    featurizer.transform(imageDF).show()
  }

  "NNModel" should "construct with sampleTransformer" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val sampleTransformer = SeqToTensor(Array(6)) -> TensorToSample()

    val nnModel = new NNModel(model)
      .setSamplePreprocessing(sampleTransformer)
      .setBatchSize(nRecords)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    assert(nnModel.transform(df).count() == nRecords)
  }

  "NNModel" should "apply with differnt params" in {
    val model = Linear[Float](6, 2)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    Seq(
      NNModel(model),
      NNModel(model, Array(6)),
      NNModel(model, SeqToTensor(Array(6)))
    ).foreach { e =>
      e.transform(df).count()
      assert(e.getBatchSize == 4 * 2)
    }
  }

  "An NNModel" should "return same results after saving and loading" in {
    val data = sqlContext.createDataFrame(smallData).toDF("features", "label")
    val module = new Sequential[Double]().add(Linear[Double](6, 2)).add(LogSoftMax[Double])
    val nnModel = NNModel(module, SeqToTensor[Double](Array(6)))

    val tmpFile = File.createTempFile("DLModel", "bigdl")
    val filePath = tmpFile.getPath + Random.nextLong().toString
    nnModel.setBatchSize(10)
      .setFeaturesCol("test123").setPredictionCol("predict123")
    nnModel.write.overwrite().save(filePath)
    val nnModel2 = try {
      NNModel.load(filePath)
    } finally {
      Path(tmpFile).deleteRecursively()
      Path(filePath).deleteRecursively()
    }

    nnModel2.uid shouldEqual nnModel.uid
    nnModel2.getBatchSize shouldEqual nnModel.getBatchSize
    nnModel2.getFeaturesCol shouldEqual nnModel.getFeaturesCol
    nnModel2.getPredictionCol shouldEqual nnModel.getPredictionCol
    nnModel2.setFeaturesCol("features").setPredictionCol("prediction")
    val result2 = nnModel2.transform(data).rdd.map(_.getSeq[Double](2).head).collect()

    nnModel.setFeaturesCol("features").setPredictionCol("prediction")
    val result = nnModel.transform(data).rdd.map(_.getSeq[Double](2).head).collect()

    result2 shouldEqual result
  }

  "An NNModel" should "support save/load keras model" in {
    val modelLocation = getClass.getClassLoader.getResource("models/nnframes_keras").getFile
    /**   code to generate model
     *  import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Embedding, Input}
     *  import com.intel.analytics.zoo.pipeline.api.keras.models.Model
     *  val input = Input(inputShape = Shape(6))
     *  val output = Dense(2).inputs(input)
     *  val module = Model(input, output)
     *  val nnModel = NNModel(module, Array(6)).setBatchSize(10)
     *  nnModel.write.overwrite().save(modelLocation)
     */
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("2")) {
      val data = sqlContext.createDataFrame(smallData).toDF("features", "label")
      NNModel.load(modelLocation).setBatchSize(8) // try load multiple times.
      val loadedModel = NNModel.load(modelLocation).setBatchSize(8)
      loadedModel.transform(data).collect()
    }
  }

  "An NNEstimator" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(2)
      .setFeaturesCol("abc")
      .setTrainSummary(new TrainSummary("/tmp", "1"))
      .setValidationSummary(new ValidationSummary("/tmp", "2"))
      .setValidation(Trigger.maxIteration(3), df, Array(new Loss[Float]()), 2)
    val copied = estimator.copy(ParamMap.empty)
    assert(estimator.model ne copied.model)
    assert(estimator.criterion ne copied.criterion)

    assert(estimator.model == copied.model)
    assert(estimator.criterion == copied.criterion)

    NNEstimatorSpec.compareParams(estimator, copied)

    val estVal = estimator.getValidation.get
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

  "An NNModel" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("abc", "la")
    val estimator = NNEstimator( model, criterion, Array(6), Array(1))
      .setBatchSize(32)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(3)
      .setFeaturesCol("abc")
      .setLabelCol("la")

    val nnModel = estimator.fit(df)
    val copied = nnModel.copy(ParamMap.empty)
    assert(nnModel.model ne copied.model)

    assert(nnModel.model == copied.model)
    NNEstimatorSpec.compareParams(nnModel, copied)
  }

  "An NNEstimator" should "work with gradient clipping" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setMaxEpoch(maxEpoch)
      .setConstantGradientClipping(-0.1f, 0.1f)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = estimator.fit(df)

    estimator.clearGradientClipping()
    estimator.fit(df)
    estimator.setGradientClippingByL2Norm(0.2f)
    estimator.fit(df)
  }

  "An NNEstimator" should "support checkpoint" in {
    val tmpFile = Files.createTempDir().getPath
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = NNEstimator(model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setMaxEpoch(5)
      .setCheckpoint(tmpFile, Trigger.everyEpoch)
    assert(estimator.getCheckpointPath == tmpFile)

    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    try {
      estimator.fit(df)
      val f = new File(tmpFile)
      assert(new File(tmpFile).listFiles().length > 0)

      // fit again to test overwrite works without error
      NNEstimator(model, criterion, Array(6), Array(1))
        .setBatchSize(nRecords)
        .setMaxEpoch(5)
        .setCheckpoint(tmpFile, Trigger.everyEpoch, isOverWrite = true).fit(df)
    } finally {
      Path(tmpFile).deleteRecursively()
    }
  }
}

private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])

object NNEstimatorSpec {
  // Generate noisy input of the form Y = signum(x.dot(weights) + intercept + noise)
  def generateTestInput(
      numRecords: Int,
      weight: Array[Double],
      intercept: Double,
      seed: Long): Seq[(Array[Double], Double)] = {
    val rnd = new Random(seed)
    val data = (1 to numRecords)
      .map( i => Array.tabulate(weight.length)(index => rnd.nextDouble() * 2 - 1))
      .map { record =>
        val y = record.zip(weight).map(t => t._1 * t._2).sum
        +intercept + 0.01 * rnd.nextGaussian()
        val label = if (y > 0) 2.0 else 1.0
        (record, label)
      }
    data
  }

  def compareParams(original: Params, copied: Params): Unit = {
    original.params.foreach { p =>
      if (original.isDefined(p)) {
        (original.getOrDefault(p), copied.getOrDefault(p)) match {
          case (Array(values), Array(newValues)) =>
            assert(values == newValues, s"Values do not match on param ${p.name}.")
          case (value, newValue) =>
            assert(value == newValue, s"Values do not match on param ${p.name}.")
        }
      } else {
        assert(!copied.isDefined(p), s"Param ${p.name} shouldn't be defined.")
      }
    }
  }
}

// F --> Sample[T]
class SeqToSample extends Preprocessing[Seq[Any], Sample[Float]] {
  override def apply(prev: Iterator[Seq[Any]]): Iterator[Sample[Float]] = {
    prev.map { case s =>
      val t = Tensor(s.length).zero()
      Sample(t, Tensor(Array(1)).fill(1f))
    }
  }
}
