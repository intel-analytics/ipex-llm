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
package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{LBFGS, Loss, SGD, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, RandomGenerator}
import com.intel.analytics.zoo.common._
import com.intel.analytics.zoo.feature.pmem.DISK_AND_DRAM
import com.intel.analytics.zoo.feature.{DistributedDataSetWrapper, DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object DistriEstimatorSpec {
  private val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 0.0, 1.0)))
  private val output1 = 0.0
  private val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 0.0, 1.0, 0.0)))
  private val output2 = 1.0
  private var plusOne = 0.0
  private val nodeNumber = 1
  private val coreNumber = 4

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

object EstimatorSpecModel {
  def mse: Module[Double] = {
    Sequential[Double]().setName("mse")
      .add(Linear[Double](4, 4).setName("fc_1"))
      .add(Sigmoid())
      .add(Linear[Double](4, 1).setName("fc_2"))
      .add(Sigmoid())
  }

  def mse2: Module[Double] = {
    Sequential[Double]()
      .add(Linear[Double](4, 8).setName("fc_1"))
      .add(Sigmoid())
      .add(Linear[Double](8, 1).setName("fc_2"))
      .add(Sigmoid())
  }

  def linear: Module[Double] = {
    Sequential[Double]
      .add(Linear(10, 5))
      .add(Sigmoid())
      .add(Linear(5, 1))
      .add(Sigmoid())
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
    Sequential[Double]
      .add(Linear(4, 2))
      .add(LogSoftMax())
  }

}

class DistriEstimatorSpec extends ZooSpecHelper {

  import DistriEstimatorSpec._
  import EstimatorSpecModel._

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  private var sc: SparkContext = _

  private var dataSet: DistributedFeatureSet[MiniBatch[Double]] = _

  override def doBefore(): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Test Estimator").setMaster("local[4]")
    sc = NNContext.initNNContext(conf)

    val rdd = sc.parallelize(1 to (256 * nodeNumber), nodeNumber)
      .map(prepareData)

    dataSet = new DistributedFeatureSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}

      override def toDistributed(): DistributedDataSet[MiniBatch[Double]] = {
        new DistributedDataSetWrapper[MiniBatch[Double]](this)
      }
    }

    plusOne = 0.0
    System.setProperty("bigdl.check.singleton", false.toString)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "Train with MSE and LBFGS" should "be good" in {
    LoggerFilter.redirectSparkInfoLogs()
    RandomGenerator.RNG.setSeed(10)
    val mseModel = mse
    val estimator = Estimator(mseModel, new LBFGS[Double]())
    estimator.train(dataSet, MSECriterion[Double](),
      endTrigger = Some(Trigger.maxIteration(1000)))

    val result1 = mseModel.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = mseModel.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  "Train with MSE and SGD" should "be trained with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm, new SGD[Double](20))
    estimator.train(dataSet, MSECriterion[Double](),
      Option(Trigger.maxEpoch(1)))

    val result1 = mm.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = mm.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  "Train with constant gradient clipping" should "be trained with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm, new SGD[Double](20))
    val resultBefore = mm.forward(input1).asInstanceOf[Tensor[Double]](Array(1))
    // make gradient nearly zero.
    estimator.setConstantGradientClipping(-1e-12, 1e-12)
    estimator.train(dataSet, MSECriterion[Double](),
      Option(Trigger.maxEpoch(1)))

    val result = mm.forward(input1).asInstanceOf[Tensor[Double]]
    result(Array(1)) should be(resultBefore +- 5e-5)
  }

  "Train with l2norm gradient clipping" should "be trained with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm, new SGD[Double](20))
    val resultBefore = mm.forward(input1).asInstanceOf[Tensor[Double]](Array(1))
    // make gradient nearly zero.
    estimator.setGradientClippingByL2Norm(1e-10)
    estimator.train(dataSet, MSECriterion[Double](),
      Option(Trigger.maxEpoch(1)))

    val result = mm.forward(input1).asInstanceOf[Tensor[Double]]
    result(Array(1)) should be(resultBefore +- 5e-5)
  }

  "Train multi times" should "be trained with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val estimator = Estimator(mm, sgd)
    val mse = MSECriterion[Double]()
    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    estimator.train(dataSet, mse, endTrigger = Some(Trigger.maxIteration(128)))
    state[Int]("neval") should be (129)
    state[Int]("epoch") should be (1)
    estimator.train(dataSet, mse, endTrigger = Some(Trigger.maxIteration(256)))
    state[Int]("neval") should be (257)
    state[Int]("epoch") should be (2)
    estimator.train(dataSet, mse, endTrigger = Some(Trigger.maxIteration(512)))
    state[Int]("neval") should be (513)
    state[Int]("epoch") should be (3)
  }

  "Evaluate" should "works with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm, new SGD[Double](20))
    estimator.train(dataSet, MSECriterion[Double](), Option(Trigger.maxEpoch(1)))
    val result = estimator.evaluate(dataSet, Array(new Loss[Double](MSECriterion[Double]())))
    result
  }

  "Estimator" should "works with model dir" in {
    LoggerFilter.redirectSparkInfoLogs()
    val tmpdir = com.google.common.io.Files.createTempDir().getPath()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm, new SGD[Double](20), tmpdir)
    estimator.train(dataSet, MSECriterion[Double](), Option(Trigger.maxEpoch(1)))
    val result = estimator.evaluate(dataSet, Array(new Loss[Double](MSECriterion[Double]())))
    result
  }

  "Estimator" should "works only evaluate" in {
    LoggerFilter.redirectSparkInfoLogs()
    val mm = mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val estimator = Estimator(mm)
    val result = estimator.evaluate(dataSet, Array(new Loss[Double](MSECriterion[Double]())))
    result
  }

  "Estimator" should "works fine with DiskFeatureSet nslice = 2" in {
    val rdd = sc.parallelize(1 to (256 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(2)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val estimator = Estimator(mm, sgd)
    val mse = MSECriterion[Double]()
    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    estimator.train(dset, mse, endTrigger = Some(MaxEpoch(2)))

    state[Int]("epoch") should be (3)
  }

  "Estimator" should "works fine with DiskFeatureSet nslice = 5" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(5)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    estimator.train(dset, mse, endTrigger = Some(MaxEpoch(2)))

    state[Int]("epoch") should be (3)
  }

  "Estimator" should "stop with MaxIteration" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(5)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    estimator.train(dset, mse, endTrigger = Some(MaxIteration(200)))

    state[Int]("neval") should be (201)
    state[Int]("epoch") should be (2)
  }

  "Estimator" should "stop with Or Trigger" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(5)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    estimator.train(dset, mse,
      endTrigger = Some(Or(MaxIteration(200), MinLoss(0.05f))))

    state[Int]("neval") should be <= 201
    state[Int]("epoch") should be <= 2
  }

  "Estimator" should "works with And Trigger" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(5)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    estimator.train(dset, mse,
      endTrigger = Some(MaxIteration(200)),
      checkPointTrigger = Some(Or(SeveralIteration(80), EveryEpoch())),
      validationSet = dset,
      validationMethod = Array(new Loss(mse)))

    val state = InternalOptimizerUtil.getStateFromOptiMethod(sgd)
    state[Int]("neval") should be <= 201
    state[Int]("epoch") should be <= 2
  }

  "Estimator" should "throw exception when cache percentage == 0" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(0)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    intercept[Exception] {
      estimator.train(dset, mse, endTrigger = Some(MaxIteration(200)))
    }
  }

  "Estimator" should "evaluate should works fine" in {
    val rdd = sc.parallelize(1 to (1024 * nodeNumber), nodeNumber)
      .map(_ => Sample[Double](Tensor[Double](4).rand(), Tensor[Double](1).rand()))
    val dset = FeatureSet.rdd(rdd, DISK_AND_DRAM(0)) -> SampleToMiniBatch(batchSize = batchSize)
    val mm = EstimatorSpecModel.mse
    mm.parameters()._1.foreach(_.fill(0.125))
    val sgd = new SGD[Double](20)
    val mse = MSECriterion[Double]()
    val estimator = Estimator(mm, sgd)
    estimator.evaluate(dset, Array(new Loss[Double](mse)))
  }
}
