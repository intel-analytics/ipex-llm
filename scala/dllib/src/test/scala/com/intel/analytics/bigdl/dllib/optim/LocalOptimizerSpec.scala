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

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BGRImgToBatch, LabeledBGRImage}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.HeapData
import com.intel.analytics.bigdl.tensor.{DnnStorage, Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.visualization.TrainSummary
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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

object LocalOptimizerSpecModel {
  def creModel : Module[Float] = {
    new Sequential[Float]
      .add(new Linear(4, 2))
      .add(new LogSoftMax)
  }

  def mseModel : Module[Float] = {
    new Sequential[Float]
      .add(new Linear(4, 2))
      .add(new Sigmoid)
      .add(new Linear(2, 1))
      .add(new Sigmoid)
  }

  def mlpModel : Module[Float] = {
    Sequential[Float]()
      .add(Linear[Float](4, 4).setName("fc_1"))
      .add(Sigmoid[Float]())
      .add(Linear[Float](4, 1).setName("fc_2"))
      .add(Sigmoid[Float]())
  }

  def dnnModel: Module[Float] = {
    new nn.mkldnn.Sequential()
      .add(nn.mkldnn.Input(Array(4, 4), Memory.Format.nc))
      .add(nn.mkldnn.Linear(4, 2))
      .add(nn.mkldnn.ReorderMemory(HeapData(Array(4, 2), Memory.Format.nc)))
  }
}

@com.intel.analytics.bigdl.tags.Serial
class LocalOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter{
  import LocalOptimizerSpecModel._
  import DummyDataSet._

  private val nodeNumber = 1
  private val coreNumber = 4

  before {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  "LocalOptimizer" should "train all minibatches per epoch" in {
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

    val dataSet = DataSet.array(images)

    val batchSize = 16
    val toTensor = new BGRImgToBatch(batchSize)
    val nn = new Sequential[Float]()
      .add(new Reshape(Array(3 * height * width)))
      .add(new Linear(3 * height * width, numClasses))
      .add(new LogSoftMax[Float]())
    val sampleDataSet = (dataSet -> toTensor).asInstanceOf[LocalDataSet[MiniBatch[Float]]]
    val batchDataSet = DataSet.array(sampleDataSet.data(train = false).toArray)
    assert(sampleDataSet.size() == numSamples)
    assert(batchDataSet.size() == numSamples / batchSize)

    Seq(sampleDataSet, batchDataSet).foreach { dataset =>
      RandomGenerator.RNG.setSeed(10)
      val maxEpochs = 2
      val logdir = com.google.common.io.Files.createTempDir()
      val trainSummary = TrainSummary(logdir.getPath, "minibatch-test")
      val optimMethod = new SGD[Float]()
      val optimizer = new LocalOptimizer(
        nn,
        dataset,
        ClassNLLCriterion[Float]())
        .setOptimMethod(optimMethod)
        .setTrainSummary(trainSummary)
        .setEndWhen(Trigger.maxEpoch(maxEpochs))
      val model = optimizer.optimize()
      val losses = trainSummary.readScalar("Loss")
      trainSummary.close()
      val iterations = optimMethod.state[Option[Int]]("neval").get

      iterations should be (maxEpochs * (dataset.data(train = false).length / nodeNumber))
    }
  }

  it should "not train model with duplicate layers" in {
    val m = Sequential[Float]()
    val l1 = Linear[Float](2, 3)
    val l2 = Identity[Float]()
    val c = Sequential[Float]()
    m.add(l1).add(c)
    c.add(l1).add(l2)

    intercept[IllegalArgumentException] {
      val optimizer = new LocalOptimizer(
        m,
        creDataSet,
        ClassNLLCriterion[Float]()
      )
    }
  }

  it should "not set model with duplicate layers" in {
    val m = Sequential[Float]()
    val l1 = Linear[Float](2, 3)
    val l2 = Identity[Float]()
    val c = Sequential[Float]()
    m.add(l1).add(c)
    c.add(l1).add(l2)

    val optimizer = new LocalOptimizer(
      c,
      creDataSet,
      ClassNLLCriterion[Float]()
    )
    intercept[IllegalArgumentException] {
      optimizer.setModel(m)
    }
  }

  "Train model with CrossEntropy and SGD" should "be good" in {
    RandomGenerator.RNG.setSeed(1000)
    val optimizer = new LocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    )

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.toTensor[Float].max(1)._2.valueAt(1, 1) should be(1.0)
    test.toTensor[Float].max(1)._2.valueAt(1, 2) should be(2.0)
  }

  it should "be same compare to ref optimizer" in {
    RandomGenerator.RNG.setSeed(1000)
    val optimizer = new LocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    )
    val model = optimizer.optimize()
    val weight = model.getParameters()._1

    RandomGenerator.RNG.setSeed(1000)
    val optimizerRef = new RefLocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    )
    val modelRef = optimizerRef.optimize()
    val weightRef = modelRef.getParameters()._1

    weight should be(weightRef)

  }

  "Train model with MSE and SGD" should "be good" in {
    RandomGenerator.RNG.setSeed(1000)

    val optimizer = new LocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setState(T("learningRate" -> 1.0))

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.toTensor[Float].valueAt(1, 1) < 0.5 should be(true)
    test.toTensor[Float].valueAt(2, 1) > 0.5 should be(true)
  }

  it should "be same compare to ref optimizer" in {
    RandomGenerator.RNG.setSeed(1000)
    val optimizer = new LocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setState(T("learningRate" -> 1.0)).setEndWhen(Trigger.maxIteration(100))
    val model = optimizer.optimize()
    val weight = model.getParameters()._1

    RandomGenerator.RNG.setSeed(1000)
    val optimizerRef = new RefLocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setState(T("learningRate" -> 1.0)).setEndWhen(Trigger.maxIteration(100))
    val modelRef = optimizerRef.optimize()
    val weightRef = modelRef.getParameters()._1
    weight should be(weightRef)
  }

  "Train model with CrossEntropy and LBFGS" should "be good" in {
    RandomGenerator.RNG.setSeed(1000)

    val optimizer = new LocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.toTensor[Float].max(1)._2.valueAt(1, 1) should be(1.0)
    test.toTensor[Float].max(1)._2.valueAt(1, 2) should be(2.0)
  }

  "Train model with multi optimMethods" should "be good" in {
    RandomGenerator.RNG.setSeed(1000)

    val optimizer = new LocalOptimizer[Float](
      mlpModel,
      mseDataSet,
      MSECriterion[Float]()
    ).setOptimMethods(Map("fc_1" -> new LBFGS[Float](), "fc_2" -> new LBFGS[Float]()))

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.toTensor[Float].valueAt(1, 1) < 0.5 should be(true)
    test.toTensor[Float].valueAt(2, 1) > 0.5 should be(true)
  }

  it should "be same compare to ref optimizer" in {
    RandomGenerator.RNG.setSeed(1000)
    val optimizer = new LocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())
    val model = optimizer.optimize()
    val weight = model.getParameters()._1

    RandomGenerator.RNG.setSeed(1000)
    val optimizerRef = new RefLocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())
    val modelRef = optimizerRef.optimize()
    val weightRef = modelRef.getParameters()._1
    weight should be(weightRef)
  }

  "Train model with MSE and LBFGS" should "be good" in {
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new LocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.toTensor[Float].valueAt(1, 1) < 0.5 should be(true)
    test.toTensor[Float].valueAt(2, 1) > 0.5 should be(true)
  }

  it should "be same compare to ref optimizer" in {
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new LocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())
    optimizer.setValidation(Trigger.everyEpoch, mseDataSet, Array(new Top1Accuracy[Float]()))
    val model = optimizer.optimize()
    val weight = model.getParameters()._1

    RandomGenerator.RNG.setSeed(10)
    val optimizerRef = new RefLocalOptimizer[Float](
      mseModel,
      mseDataSet,
      new MSECriterion[Float].asInstanceOf[Criterion[Float]]
    ).setOptimMethod(new LBFGS[Float]())
    val modelRef = optimizerRef.optimize()
    val weightRef = modelRef.getParameters()._1
    weight should be(weightRef)
  }

  "Train model with CrossEntropy and SGD" should "be good with constant clipping" in {
    val _learningRate = 20.0
    val optimizationMethod = new SGD[Float](learningRate = _learningRate)
    val optimizer = new LocalOptimizer[Float](
      creModel,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setConstantGradientClipping(0.0, 0.0)
      .setEndWhen(Trigger.maxEpoch(1))
      .setOptimMethod(optimizationMethod)


    val model = optimizer.optimize()
    val newG = model.getParameters()._2

    assert(newG.sumSquare() == 0, "gradient should be 0")
  }

  "Train model with CrossEntropy and SGD" should "be good with l2norm clipping" in {
    RandomGenerator.RNG.setSeed(1000)
    val linear = Linear[Float](4, 2)
    val _learningRate = 0.0
    val optimizationMethod = new SGD[Float](learningRate = _learningRate)
    val optimizer = new LocalOptimizer[Float](
      linear,
      creDataSet,
      new ClassNLLCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setEndWhen(Trigger.maxIteration(1))
      .setOptimMethod(optimizationMethod)

    val model = optimizer.optimize()
    val gradient = model.getParameters()._2.clone()
    val scale = math.sqrt(gradient.sumSquare()) / 0.03
    val expectedG = gradient.clone().div(scale.toFloat)

    val optimizationMethod2 = new SGD[Float](learningRate = _learningRate)
    linear.getParameters()._1.fill(2.5f)
    val optimizer2 = new LocalOptimizer[Float](linear, creDataSet,
      new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Float]])
      .setEndWhen(Trigger.maxIteration(1))
      .setOptimMethod(optimizationMethod2)
      .setGradientClippingByl2Norm(0.03)

    val model2 = optimizer2.optimize()
    val newG = model2.getParameters()._2
    assert(expectedG.almostEqual(newG, 0.0), "clipbynorm2 should generate correct gradient")
  }
}
