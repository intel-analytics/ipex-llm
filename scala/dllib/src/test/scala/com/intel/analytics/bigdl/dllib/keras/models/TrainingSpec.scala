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

package com.intel.analytics.zoo.pipeline.api.keras.models

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.optim.{SGD, Top1Accuracy}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.pipeline.api.autograd.{Variable, AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.python.PythonZooKeras
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.apache.commons.io.FileUtils

import scala.reflect.ClassTag

class TrainingSpec extends ZooSpecHelper {

  private var sc: SparkContext = _

  override def doBefore(): Unit = {
    val conf = new SparkConf()
      .setMaster("local[4]")
    sc = NNContext.initNNContext(conf, appName = "TrainingSpec")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  def generateData(featureShape: Array[Int], labelSize: Int, dataSize: Int): RDD[Sample[Float]] = {
    sc.range(0, dataSize, 1).map { _ =>
      val featureTensor = Tensor[Float](featureShape)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](labelSize)
      labelTensor(Array(labelSize)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }
  }

  "initNNContext" should "contain spark-analytics-zoo.conf properties" in {
    sc = SparkContext.getOrCreate()
    sc.stop()
    val conf = new SparkConf()
      .setMaster("local[4]")
    sc = NNContext.initNNContext(conf, "hello")
    assert(sc.appName == "hello")
    sc.getConf.get("spark.serializer") should be
    ("org.apache.spark.serializer.JavaSerializer")
    sc.getConf.get("spark.scheduler.minRegisteredResourcesRatio") should be ("1.0")
  }

  "sequential compile and fit with custom loss" should "work properly" in {
    val trainingData = generateData(Array(10), 5, 40)
    val model = Sequential[Float]()
    model.add(Dense[Float](5, inputShape = Shape(10)))
    def cLoss[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
      implicit ev: TensorNumeric[T]): Variable[T] = {
      A.mean(A.abs(yTrue - yPred), axis = 1)
    }
    model.compile(optimizer = new SGD[Float](), loss = cLoss[Float] _)
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)
  }

  "graph compile and fit" should "work properly" in {
    val trainingData = generateData(Array(10), 1, 40)
    val input = Input[Float](inputShape = Shape(10))
    val output = Dense[Float](1, activation = "relu").inputs(input)
    val model = Model[Float](input, output)
    model.compile(optimizer = "adam", loss = "mae", metrics = List("auc"))
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)
  }

  "compile, fit with validation, evaluate, predict, setTensorBoard, " +
    "setCheckPoint, gradientClipping" should "work properly" in {
    val trainingData = generateData(Array(12, 12), 1, 100)
    val testData = generateData(Array(12, 12), 1, 16)
    val model = Sequential[Float]()
    model.add(Dense[Float](8, activation = "relu", inputShape = Shape(12, 12)))
    model.add(Flatten[Float]())
    model.add(Dense[Float](2, activation = "softmax"))
    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    val tmpLogDir = createTmpDir()
    val tmpCheckpointDir = createTmpDir()
    model.setTensorBoard(tmpLogDir.getAbsolutePath, "TrainingSpec")
    model.setCheckpoint(tmpCheckpointDir.getAbsolutePath)
    model.setGradientClippingByL2Norm(0.2f)
    model.fit(trainingData, batchSize = 8, validationData = testData, nbEpoch = 2)
    model.clearGradientClipping()
    model.fit(trainingData, batchSize = 8, validationData = testData, nbEpoch = 2)
    model.setGradientClippingByL2Norm(0.2f)
    model.fit(trainingData, batchSize = 8, validationData = testData, nbEpoch = 2)
    val accuracy = model.evaluate(testData, batchSize = 8)
    val predictResults = model.predict(testData, batchPerThread = 8)
    FileUtils.deleteDirectory(tmpLogDir)
    FileUtils.deleteDirectory(tmpCheckpointDir)
  }

  "compile, fit, evaluate and predict in local mode" should "work properly" in {
    val localData = DummyDataSet.mseDataSet
    val model = Sequential[Float]()
    model.add(Dense[Float](8, activation = "relu", inputShape = Shape(4)))
    model.compile(optimizer = new SGD[Float](), loss = MSECriterion[Float](),
      metrics = List(new Top1Accuracy[Float]))
    model.setConstantGradientClipping(0.01f, 0.03f)
    model.fit(localData, nbEpoch = 2)
    model.clearGradientClipping()
    model.fit(localData, nbEpoch = 2)
    val accuracy = model.evaluate(localData)
    val predictResults = model.predict(localData, 32)
  }

  "model predictClass giving zero-based label" should "work properly" in {
    val data = new Array[Sample[Float]](100)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](28, 28, 1).rand()
      val label = Tensor[Float](1).fill(0.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = Sequential[Float]()
    model.add(Flatten[Float](inputShape = Shape(28, 28, 1)))
    model.add(Dense[Float](10, activation = "softmax"))
    val dataSet = sc.parallelize(data, 2)
    val result = model.predictClasses(dataSet)

    val prob = result.collect()
    prob.zip(data).foreach(item => {
      val res = model.forward(item._2.feature.reshape(Array(1, 28, 28, 1)))
        .toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt
      (res-1) should be (item._1)
    })
  }

  "fit, predict and evaluate on ImageSet" should "work properly" in {

    def createImageFeature(): ImageFeature = {
      val feature = new ImageFeature()
      val data = Tensor[Float](200, 200, 3).rand()
      val mat = OpenCVMat.fromFloats(data.storage.toArray, 200, 200, 3)
      feature(ImageFeature.bytes) = OpenCVMat.imencode(mat)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextInt(20))
      feature(ImageFeature.label) = labelTensor
      feature
    }

    def createImageSet(dataSize: Int): ImageSet = {
      val rdd = sc.range(0, dataSize, 1).map { _ =>
        createImageFeature()
      }
      ImageSet.rdd(rdd)
    }

    val trainingData = createImageSet(64)
    val testData = createImageSet(16)
    val transformer = ImageBytesToMat() -> ImageResize(256, 256) ->
      ImageCenterCrop(224, 224) -> ImageMatToTensor[Float]() ->
      ImageSetToSample[Float](targetKeys = Array("label"))
    trainingData.transform(transformer)
    testData.transform(transformer)
    val model = Sequential[Float]()
    model.add(Convolution2D[Float](1, 5, 5, inputShape = Shape(3, 224, 224)))
    model.add(MaxPooling2D[Float]())
    model.add(Flatten[Float]())
    model.add(Dense[Float](20, activation = "softmax"))
    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    model.fit(trainingData, nbEpoch = 2, batchSize = 8, validationData = testData)
    model.predict(testData, batchPerThread = 8)
    val accuracy = model.evaluate(testData, batchSize = 8)
  }

  "zooEvaluate" should "work" in {
    val trainingData = generateData(Array(12, 12), 1, 100)
    val model = Sequential[Float]()

    model.add(Dense[Float](8, activation = "relu", inputShape = Shape(12, 12)))
    model.add(Flatten[Float]())
    model.add(Dense[Float](2, activation = "softmax"))

    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)

    val api = new PythonZooKeras[Float]()
    val bigdlApi = sc.broadcast(new PythonBigDL[Float]())

    // python api require to take no type Sample and it takes JavaRDD as input
    // use toPySample to convert to no type use toJavaRDD to convert to JavaRDD
    val jd = trainingData.map(j => bigdlApi.value.toPySample(j)).toJavaRDD()
    val res = api.zooEvaluate(model, jd, 8)
    res
  }

  "tensorboard api" should "work" in {
    val trainingData = generateData(Array(12, 12), 1, 100)
    val model = Sequential[Float]()

    model.add(Dense[Float](8, activation = "relu", inputShape = Shape(12, 12)))
    model.add(Flatten[Float]())
    model.add(Dense[Float](2, activation = "softmax"))

    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    val api = new PythonZooKeras[Float]()

    model.setTensorBoard("./", "testTensorBoard")
    model.fit(trainingData, batchSize = 8, nbEpoch = 2, validationData = trainingData)

    val rawTrain = model.getTrainSummary("Loss")
    val rawVal = model.getValidationSummary("Loss")

    val trainArr = api.zooGetScalarFromSummary(model, "Loss", "Train")
    val valArr = api.zooGetScalarFromSummary(model, "Loss", "Validation")

    // delete test directory
    import scala.reflect.io.Directory
    import java.io.File
    val dir = new Directory(new File("./testTensorBoard"))
    if (dir.exists && dir.isDirectory) {
      dir.deleteRecursively()
    }
    valArr
  }
}

object DummyDataSet extends LocalDataSet[MiniBatch[Float]] {
  val totalSize = 10
  var isCrossEntropy = true

  def creDataSet: LocalDataSet[MiniBatch[Float]] = {
    isCrossEntropy = true
    DummyDataSet
  }

  def mseDataSet: LocalDataSet[MiniBatch[Float]] = {
    isCrossEntropy = false
    DummyDataSet
  }

  private val feature = Tensor[Float](
    Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0
      )
    ),
    storageOffset = 1,
    size = Array(4, 4)
  )
  private val labelMSE = Tensor[Float](
    Storage[Float](
      Array[Float](
        0,
        1,
        0,
        1
      )
    ),
    storageOffset = 1,
    size = Array(4)
  )

  private val labelCrossEntropy = Tensor[Float](
    Storage[Float](
      Array[Float](
        1,
        2,
        1,
        2
      )
    ),
    storageOffset = 1,
    size = Array(4)
  )

  override def size(): Long = totalSize

  override def shuffle(): Unit = {}

  override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      var i = 0

      override def hasNext: Boolean = train || i < totalSize

      override def next(): MiniBatch[Float] = {
        i += 1
        MiniBatch(feature, if (isCrossEntropy) labelCrossEntropy else labelMSE)
      }
    }
  }
}
