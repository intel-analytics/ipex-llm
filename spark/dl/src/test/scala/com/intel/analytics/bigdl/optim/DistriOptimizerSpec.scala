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

package com.intel.analytics.bigdl.optim

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BGRImgToBatch, LabeledBGRImage}
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.tensor.{DenseTensor, Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.TrainSummary
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object DistriOptimizerSpec {
  private val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 0.0, 1.0)))
  private val output1 = 0.0
  private val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 0.0, 1.0, 0.0)))
  private val output2 = 1.0
  private var plusOne = 0.0
  private val nodeNumber = 4
  private val coreNumber = 4
  Engine.init(nodeNumber, coreNumber, onSpark = true)

  private val batchSize = 2 * coreNumber

  private val prepareData: Int => (MiniBatch[Double]) = index => {
    val input = Tensor[Double]().resize(batchSize, 4)
    val target = Tensor[Double]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, output1 + plusOne)
        input.select(1, i + 1).copy(input1)
      } else {
        target.setValue(i + 1, output2 + plusOne)
        input.select(1, i + 1).copy(input2)
      }
      i += 1
    }
    MiniBatch(input, target)
  }
}

object DistriOptimizerSpecModel {
  def mse: Module[Double] = {
    new Sequential[Double]
      .add(new Linear(4, 2))
      .add(new Sigmoid)
      .add(new Linear(2, 1))
      .add(new Sigmoid)
  }

  def bn: Module[Double] = {
    Sequential[Double]
      .add(Linear(4, 2))
      .add(BatchNormalization(2))
      .add(ReLU())
      .add(Linear(2, 1))
      .add(Sigmoid())
  }

  def cre: Module[Double] = {
    new Sequential[Double]
      .add(new Linear(4, 2))
      .add(new LogSoftMax)
  }

  def mserf(failCountNumberLists: Array[Int], sleep: Boolean = false): Module[Double] = {
    new Sequential[Double]
      .add(new Linear(4, 2))
      .add(new Sigmoid)
      .add(new Linear(2, 1))
      .add(new Sigmoid)
      .add(new ExceptionTest(failCountNumberLists, sleep))
  }
}

@com.intel.analytics.bigdl.tags.Serial
class DistriOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  import DistriOptimizerSpec._
  import DistriOptimizerSpecModel._

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  private var sc: SparkContext = _

  private var dataSet: DistributedDataSet[MiniBatch[Double]] = _

  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * nodeNumber), nodeNumber).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }

    plusOne = 0.0
    System.setProperty("bigdl.check.singleton", false.toString)
    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "DistriOptimizer" should "train all minibatches per epoch" in {
    val numSamples = 64
    val numClasses = 3
    val height = 32
    val width = 32
    val images = Array.tabulate(64) { i =>
      val image = new LabeledBGRImage(width, height)
      image.setLabel((i % numClasses).toFloat + 1F)
      val tensor = Tensor[Float](Storage[Float](image.content), 1, Array(3, width, height))
      tensor.rand()
      image
    }

    val numPartitions = 4
    val dataSet = DataSet.rdd(sc.parallelize(images, numPartitions))

    val batchSize = 16
    val toTensor = new BGRImgToBatch(batchSize)
    val nn = new Sequential[Float]()
      .add(new Reshape(Array(3 * height * width)))
      .add(new Linear(3 * height * width, numClasses))
      .add(new LogSoftMax[Float]())
    val sampleDataSet = (dataSet -> toTensor).asInstanceOf[DistributedDataSet[MiniBatch[Float]]]
    val batchDataSet = DataSet.rdd(sampleDataSet.data(train = false))
    assert(sampleDataSet.size() == numSamples)
    assert(batchDataSet.size() == numSamples / batchSize * numPartitions)

    Seq(sampleDataSet, batchDataSet).foreach { dataset =>
      RandomGenerator.RNG.setSeed(10)
      val maxEpochs = 2
      val logdir = com.google.common.io.Files.createTempDir()
      val trainSummary = TrainSummary(logdir.getPath, "minibatch-test")
      val optimizer = new DistriOptimizer(
        nn,
        dataset,
        ClassNLLCriterion[Float]())
        .setOptimMethod(new LBFGS)
        .setTrainSummary(trainSummary)
        .setEndWhen(Trigger.maxEpoch(maxEpochs))
      val model = optimizer.optimize()
      val losses = trainSummary.readScalar("Loss")
      trainSummary.close()

      losses should have length maxEpochs * (dataset.data(train = false).count() / nodeNumber)
    }
  }

  it should "not train model with duplicate layers" in {
    val m = Sequential[Double]()
    val l1 = Identity[Double]()
    val l2 = Identity[Double]()
    val c = Sequential[Double]()
    m.add(l1).add(c)
    c.add(l1).add(l2)

    intercept[IllegalArgumentException] {
      val optimizer = new DistriOptimizer[Double](
        m,
        dataSet,
        ClassNLLCriterion[Double]()
      )
    }
  }

  it should "not set model with duplicate layers" in {
    val m = Sequential[Double]()
    val l1 = Identity[Double]()
    val l2 = Identity[Double]()
    val c = Sequential[Double]()
    m.add(l1).add(c)
    c.add(l1).add(l2)

    val optimizer = new DistriOptimizer[Double](
      c,
      dataSet,
      ClassNLLCriterion[Double]()
    )
    intercept[IllegalArgumentException] {
      val optimizer = new DistriOptimizer[Double](
        m,
        dataSet,
        ClassNLLCriterion[Double]()
      )
    }
  }

  "Train with MSE and LBFGS" should "be good" in {
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer(
      mse,
      dataSet,
      new MSECriterion[Double]())
      .setOptimMethod(new LBFGS)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  "Train with MSE and SGD" should "be trained with good result" in {
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  "Train with MSE and SGD" should "be trained with good result after reset model" in {
    var mm = bn
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
    optimizer.optimize()

    mm = mse
    mm.getParameters()._1.fill(0.125)
    optimizer.setModel(mm)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  it should "be same compare to ref optimizer" in {
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer(
      mse,
      dataSet,
      new MSECriterion[Double]())
    val model = optimizer.optimize()

    RandomGenerator.RNG.setSeed(10)
    val optimizerRef = new RefDistriOptimizer(
      mse,
      dataSet,
      new MSECriterion[Double]()
    )
    val modelRef = optimizerRef.optimize()

    model.getParameters()._1 should be(modelRef.getParameters()._1)
  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should
    "be trained with good result" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](cre, dataSet,
      new ClassNLLCriterion[Double]())
      .setEndWhen(Trigger.maxEpoch(1)).setOptimMethod(new LBFGS)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should
    "be trained with good result" in {
    plusOne = 1.0
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer[Double](cre, dataSet,
      new ClassNLLCriterion[Double]())
      .setState(T("learningRate" -> 20.0))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be same compare to ref optimizer" in {
    plusOne = 1.0
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer[Double](
      cre,
      dataSet,
      new ClassNLLCriterion[Double]()
    ).setState(T("learningRate" -> 20.0))
    val model = optimizer.optimize()

    RandomGenerator.RNG.setSeed(10)
    val optimizerRef = new RefDistriOptimizer(
      cre,
      dataSet,
      new ClassNLLCriterion[Double]()
    ).setState(T("learningRate" -> 20.0))
    val modelRef = optimizerRef.optimize()

    model.getParameters()._1 should be(modelRef.getParameters()._1)

  }

  "Train with BatchNormalization" should "return with state" in {
    RandomGenerator.RNG.setSeed(10)
    val mm = bn
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
    val model = optimizer.optimize()
    val batchNormalization = model.asInstanceOf[Sequential[Double]].modules(1).
      asInstanceOf[BatchNormalization[Double]]
    val expectedMeans = Array(0.37499998210083496, 0.37499998210083496)
    val expectedVariances = Array(1188.2811870277535, 1188.2811870277535)
    batchNormalization.runningMean.storage().array().zip(expectedMeans).foreach {
      case (actual, expected) => actual should be(expected +- 1e-4)
    }
    batchNormalization.runningVar.storage().array().zip(expectedVariances).foreach {
      case (actual, expected) => actual should be(expected +- 1e-4)
    }
  }

  "Train with one partition one executor" should "won't throw mult-task exception" in {
    System.setProperty("bigdl.check.singleton", true.toString)
    RandomGenerator.RNG.setSeed(10)
    Engine.setNodeNumber(1)
    val mm = bn
    mm.getParameters()._1.fill(0.125)
    val rdd = sc.parallelize(1 to (256 * nodeNumber), 1).map(prepareData)
    val dataSet = new DistributedDataSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = 256 * nodeNumber

      override def shuffle(): Unit = {}
    }
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
      .optimize()

    Engine.setNodeNumber(nodeNumber)
  }

  "DistriOptimizer checkpoint" should "work correctly" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))

    import com.intel.analytics.bigdl._
    plusOne = 1.0
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer[Double](
      cre,
      dataSet,
      new ClassNLLCriterion[Double]()
    )
    optimizer.setState(T("learningRate" -> 20.0))
      .setCheckpoint(filePath, Trigger.everyEpoch)
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()

    val numIterations = dataSet.data(train = false).count() / nodeNumber + 1
    val optimMethod = OptimMethod.load[Double](optimizer.getCheckpointPath().get +
      s"/optimMethod.$numIterations")

    optimMethod.state.get[Int]("epoch").get should be(2)
    optimMethod.state.get[Int]("neval").get should be(numIterations)
  }

  "TrainSummary with MSE and LBFGS" should "work correctly" in {
    TestUtils.cancelOnWindows()
    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val trainSummary = TrainSummary(logdir.getPath, "lbfgs")
    val optimizer = new DistriOptimizer(
      mse,
      dataSet,
      new MSECriterion[Double]())
      .setOptimMethod(new LBFGS)
      .setTrainSummary(trainSummary)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 1e-2)
    trainSummary.readScalar("Loss").last._2 should be(0.0f +- 1e-3f)
    trainSummary.close()
  }

  "TrainSummary with MSE and SGD" should "work correctly" in {
    TestUtils.cancelOnWindows()
    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val trainSummary = TrainSummary(logdir.getPath, "sgd")
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
      .setTrainSummary(trainSummary)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
    trainSummary.readScalar("Loss").last._2 should be(0.0f +- 1e-3f)
    trainSummary.close()
  }

  "TrainSummary with MSE and Adagrad" should "work correctly" in {
    TestUtils.cancelOnWindows()
    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val trainSummary = TrainSummary(logdir.getPath, "adagrad")
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 1.0))
      .setOptimMethod(new Adagrad[Double]())
      .setEndWhen(Trigger.maxEpoch(1))
      .setTrainSummary(trainSummary)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
    trainSummary.readScalar("Loss").last._2 should be(0.0f +- 1e-3f)
    trainSummary.close()
  }

  "Train with MSE and SGD" should "be trained with good result with failures in small interval" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val failCountNumberList = Array(800, 850, 900)
    val mm = mserf(failCountNumberList)
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
      .setCheckpoint(filePath, Trigger.everyEpoch)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)

    ExceptionTest.resetCount()
  }

  "Train with MSE and SGD" should "be trained with good result with failures in big interval" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val failCountNumberList = Array(800, 850, 900, 1500)
    System.setProperty("bigdl.failure.retryTimeInterval", "3")
    System.setProperty("bigdl.failure.retryTimes", "2")
    val mm = mserf(failCountNumberList, true)
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
      .setCheckpoint(filePath, Trigger.everyEpoch)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)

    ExceptionTest.resetCount()
  }

  "Train with MSE and SGD" should "throw exception after retry times exceed settings" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    Files.delete(Paths.get(filePath))
    Files.createDirectory(Paths.get(filePath))
    val failCountNumberList = Array(800, 850, 900)
    System.setProperty("bigdl.failure.retryTimes", "3")
    val mm = mserf(failCountNumberList)
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))

    intercept[Exception] {
      optimizer.optimize()
    }
    ExceptionTest.resetCount()

    optimizer.setCheckpoint(filePath, Trigger.everyEpoch)
    intercept[Exception] {
      optimizer.optimize()
    }
    ExceptionTest.resetCount()
  }

  "Train with Plateau" should "work properly" in {
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
    Logger.getLogger("com.intel.analytics.bigdl").setLevel(Level.INFO)

    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](
      _model = mm,
      _dataset = dataSet,
      _criterion = new MSECriterion[Double]()
    )

    val optimMethod = new SGD[Double](learningRate = 20.0, learningRateSchedule =
      SGD.Plateau("Loss", epsilon = 0, patience = 1, mode = "min"))

    optimizer.setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(1))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  "Train with Plateau Score" should "work properly" in {
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
    Logger.getLogger("com.intel.analytics.bigdl").setLevel(Level.INFO)

    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](
      _model = mm,
      _dataset = dataSet,
      _criterion = new MSECriterion[Double]()
    )

    val optimMethod = new SGD[Double](learningRate = 20.0, learningRateSchedule =
      SGD.Plateau("score", epsilon = 0, patience = 1, mode = "max"))

    optimizer.setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(1))
    optimizer.setValidation(Trigger.everyEpoch, dataSet,
      Array(new Top1Accuracy[Double]()))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  "Train with L1Regularization" should "work properly in DistriOptimizer" in {
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
    Logger.getLogger("com.intel.analytics.bigdl").setLevel(Level.INFO)

    RandomGenerator.RNG.setSeed(10)
    val logdir = com.google.common.io.Files.createTempDir()
    val mm = Sequential[Double]().add(Linear(4, 2,
      wRegularizer = L1Regularizer(1), bRegularizer = L1Regularizer(1)))
      .add(Sigmoid())
      .add(Linear(2, 1))
      .add(Sigmoid())
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](
      _model = mm,
      _dataset = dataSet,
      _criterion = new MSECriterion[Double]()
    )

    val optimMethod = new SGD[Double](learningRate = 20.0)

    optimizer.setOptimMethod(optimMethod)
      .setEndWhen(Trigger.severalIteration(10))
    optimizer.setValidation(Trigger.everyEpoch, dataSet,
      Array(new Top1Accuracy[Double]()))
    val model = optimizer.optimize()
  }

  "setTrainData" should "work properly" in {

    RandomGenerator.RNG.setSeed(10)
    val rdd = sc.parallelize(1 to (2 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](2, 3).fill(2.0), Tensor[Double](1).fill(1.0)))

    val inputOri = rdd.map{s => s.feature}
    val targetOri = rdd.map{s => s.label}
    val inputOriArr = inputOri.collect()
    val targetOriArr = targetOri.collect()


    val myOpt = new DistriOptimizer[Double](Identity[Double](), dataSet, null) {
      override def optimize(): Module[Double] = {
        val dds = this.dataset.asInstanceOf[DistributedDataSet[MiniBatch[Double]]]
        val rdd = dds.data(train = false)
        // flatmap to break minibatches into single tensors
        val input = rdd.flatMap[Tensor[Double]]{
          data => data.getInput().asInstanceOf[Tensor[Double]].split(dim = 1)}
        val target = rdd.flatMap[Tensor[Double]]{
          data => data.getTarget().asInstanceOf[Tensor[Double]].split(dim = 1)}
        val inputArr = input.collect()
        val targetArr = target.collect()

        inputArr.sameElements(inputOriArr) should be (true)
        targetArr.sameElements(targetOriArr) should be (true)

        model
      }
    }

    myOpt.setTrainData(rdd, 2*nodeNumber)
    myOpt.optimize()
  }

  "Train with MSE and LarsSGD" should "be trained with good result in 1 iteration" in {
    val _learningRate = 20.0
    val _weightDecay = 0.1
    val _momentum = 0.05
    val _gwRation = 0.8

    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val oriW = mm.parameters()._1
    // update below parameters if mse model has been changed
    val layerNum = 4
    val layer_0Num = 8
    val layer_1Num = 2
    val layer_2Num = 2
    val layer_3Num = 1
    require(mm.parameters()._1.length == layerNum && mm.parameters()._1(0).nElement() == layer_0Num
      && mm.parameters()._1(1).nElement() == layer_1Num &&
      mm.parameters()._1(2).nElement() == layer_2Num &&
      mm.parameters()._1(3).nElement() == layer_3Num,
      "if mse model has been chaged, this test case must needed updated accordingly!")

    val wNorm2 = oriW.map(x => math.sqrt(x.sumSquare()))

    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxIteration(1))
      .setOptimMethod(new LarsSGD[Double](
        learningRate = _learningRate,
        learningRateDecay = 0.0,
        weightDecay = _weightDecay,
        momentum = _momentum,
        gwRation = _gwRation))

    val model = optimizer.optimize()
    val newW = model.parameters()._1.clone()
    val newG = model.parameters()._2.clone()

    val gNorm2 = newG.map(x => math.sqrt(x.sumSquare()))
    var i = 0
    while (i < wNorm2.length) {
      val clr = _learningRate * (wNorm2(i) / (gNorm2(i) + _weightDecay * wNorm2(i))) * _gwRation
      newG(i).add(_weightDecay, oriW(i))
      val expectW = oriW(i).add(-1.0, newG(i).mul(clr))
      assert(expectW.almostEqual(newW(i), 1e-6) == true, "should generate correct weight")
      i += 1
    }
  }

  "Train with MSE and LarsSGD" should "be be able to train with more than 1 iteration" in {
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    // update below parameters if mse model has been changed
    val layerNum = 4
    val layer_0Num = 8
    val layer_1Num = 2
    val layer_2Num = 2
    val layer_3Num = 1
    require(mm.parameters()._1.length == layerNum && mm.parameters()._1(0).nElement() == layer_0Num
      && mm.parameters()._1(1).nElement() == layer_1Num &&
      mm.parameters()._1(2).nElement() == layer_2Num &&
      mm.parameters()._1(3).nElement() == layer_3Num,
      "if mse model has been chaged, this test case must needed updated accordingly!")

    val _learningRate = 20.0
    val _weightDecay = 0.1
    val _momentum = 0.05
    val _gwRation = 0.8
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxEpoch(1))
      .setOptimMethod(new LarsSGD[Double](
        learningRate = _learningRate,
        learningRateDecay = 0.0,
        weightDecay = _weightDecay,
        momentum = _momentum,
        gwRation = _gwRation))

    val model = optimizer.optimize()
  }

  "Train with MSE " should "generate correct gradients with constant clipping" in {
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val oriW = mm.getParameters()._1.clone()

    val _learningRate = 20.0
    val optimizationMethod = new SGD[Double](learningRate = _learningRate)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxEpoch(1))
      .setOptimMethod(optimizationMethod)
      .setConstantGradientClipping(-0.0f, 0.0f)

    val model = optimizer.optimize()
    val newW = model.getParameters()._1
    val newG = model.getParameters()._2

    assert(newW.almostEqual(oriW, 0.0) == true, "weight should keep the same")
    assert(newG.almostEqual(oriW.fill(0.0), 0.0) == true, "gradient should be 0")
  }

  "Train with MSE " should "generate correct gradients with l2norm clipping" in {
    val mm = mse
    mm.getParameters()._1.fill(0.125)

    val _learningRate = 20.0
    val optimizationMethod = new SGD[Double](learningRate = _learningRate)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxIteration(1))
      .setOptimMethod(optimizationMethod)

    val model = optimizer.optimize()
    val gradient = model.getParameters()._2.clone()
    val scale = math.sqrt(gradient.sumSquare()) / 0.03
    val expectedG = gradient.clone().div(scale)

    val mm2 = mse
    mm2.getParameters()._1.fill(0.125)
    val optimizationMethod2 = new SGD[Double](learningRate = _learningRate)
    val optimizer2 = new DistriOptimizer[Double](mm2, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxIteration(1))
      .setOptimMethod(optimizationMethod2)
      .setGradientClippingByl2Norm(0.03)

    val model2 = optimizer2.optimize()
    val newG = model2.getParameters()._2
    assert(expectedG.almostEqual(newG, 0.0), "clipbynorm2 should generate correct gradient")
  }

  "optimMethod state " should "be updated correctly after optimize" in {
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
    Logger.getLogger("com.intel.analytics.bigdl").setLevel(Level.INFO)

    val mm = Sequential[Double]().add(Linear(4, 1))
      .add(Sigmoid())

    val optimizer = new DistriOptimizer[Double](
      _model = mm,
      _dataset = dataSet,
      _criterion = new MSECriterion[Double]()
    )

    val optimMethod = new SGD[Double](learningRate = 20.0)

    optimizer.setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxIteration(10))
    val model = optimizer.optimize()

    optimMethod.state[Int]("epoch") should be(1)
    optimMethod.state[Int]("neval") should be(11)
    optimMethod.state[Int]("recordsProcessedThisEpoch") should be(320)

    optimizer.setEndWhen(Trigger.maxIteration(20))
    optimizer.optimize()

    optimMethod.state[Int]("epoch") should be(1)
    optimMethod.state[Int]("neval") should be(21)
    optimMethod.state[Int]("recordsProcessedThisEpoch") should be(640)

    val rdd = sc.parallelize(1 to (160 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).fill(2.0), Tensor[Double](1).fill(1.0)))

    optimizer.setTrainData(rdd, 16 * nodeNumber)

    optimMethod.state[Int]("epoch") should be(2)
    optimMethod.state[Int]("neval") should be(21)
    optimMethod.state[Int]("recordsProcessedThisEpoch") should be(0)

    optimizer.setEndWhen(Trigger.maxEpoch(2))
    optimizer.optimize()

    optimMethod.state[Int]("epoch") should be(3)
    optimMethod.state[Int]("neval") should be(31)
    optimMethod.state[Int]("recordsProcessedThisEpoch") should be(0)
  }
}


