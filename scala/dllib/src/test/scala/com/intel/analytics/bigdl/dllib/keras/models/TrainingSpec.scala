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

import com.google.common.io.Files
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.optim.{SGD, Top1Accuracy}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.apache.commons.io.FileUtils

class TrainingSpec extends FlatSpec with Matchers with BeforeAndAfter  {

  private var sc: SparkContext = _

  def generateData(shape: Array[Int], size: Int): RDD[Sample[Float]] = {
    sc.range(0, size, 1).map { _ =>
      val featureTensor = Tensor[Float](shape)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat()) + 1
      Sample[Float](featureTensor, labelTensor)
    }
  }

  before {
    val conf = new SparkConf()
      .setAppName("TrainingSpec")
      .setMaster("local[4]")
    sc = NNContext.getNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "sequential compile and fit" should "work properly" in {
    val trainingData = generateData(Array(10), 40)
    val model = Sequential[Float]()
    model.add(Dense(8, inputShape = Shape(10)))
    model.compile(optimizer = "sgd", loss = "mse", metrics = null)
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)
  }

  "graph compile and fit" should "work properly" in {
    val trainingData = generateData(Array(10), 40)
    val input = Input[Float](inputShape = Shape(10))
    val output = Dense[Float](8, activation = "relu").inputs(input)
    val model = Model[Float](input, output)
    model.compile(optimizer = "adam", loss = "mae", metrics = null)
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)
  }

  "compile, fit with validation, evaluate, predict, setTensorBoard, setCheckPoint" should
    "work properly" in {
    val trainingData = generateData(Array(12, 12), 100)
    val testData = generateData(Array(12, 12), 16)
    val model = Sequential[Float]()
    model.add(Dense(8, activation = "relu", inputShape = Shape(12, 12)))
    model.add(Flatten())
    model.add(Dense(2, activation = "softmax"))
    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    val tmpLogDir = Files.createTempDir()
    val tmpCheckpointDir = Files.createTempDir()
    model.setTensorBoard(tmpLogDir.getAbsolutePath, "TrainingSpec")
    model.setCheckpoint(tmpCheckpointDir.getAbsolutePath)
    model.fit(trainingData, batchSize = 8, validationData = testData, nbEpoch = 2)
    val accuracy = model.evaluate(testData, batchSize = 8)
    val predictResults = model.predict(testData, batchSize = 8)
    FileUtils.deleteDirectory(tmpLogDir)
    FileUtils.deleteDirectory(tmpCheckpointDir)
  }

  "compile, fit, evaluate and predict in local mode" should "work properly" in {
    val localData = DummyDataSet.mseDataSet
    val model = Sequential[Float]()
    model.add(Dense(8, activation = "relu", inputShape = Shape(4)))
    model.compile(optimizer = new SGD[Float](), loss = MSECriterion[Float](),
      metrics = List(new Top1Accuracy[Float]))
    model.fit(localData, nbEpoch = 2, validationData = null)
    val accuracy = model.evaluate(localData)
    val predictResults = model.predict(localData)
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
