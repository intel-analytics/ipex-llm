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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
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

      override def hasNext: Boolean = true

      override def next(): MiniBatch[Float] = {
        i += 1
        MiniBatch(feature, if (isCrossEntropy) labelCrossEntropy else labelMSE)
      }
    }
  }
}

object LocalOptimizerSpecModel {
  def creModel : Module[Float] = {
    val mlp = new Sequential[Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)
    mlp
  }

  def mseModel : Module[Float] = {
    val mlp = new Sequential[Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)
    mlp
  }
}

@com.intel.analytics.bigdl.tags.Serial
class LocalOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter{
  import LocalOptimizerSpecModel._
  import DummyDataSet._

  val nodeNumber = 1
  val coreNumber = 4

  before {
    Engine.localMode = true
    Engine.init(nodeNumber, coreNumber, false)
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
}
