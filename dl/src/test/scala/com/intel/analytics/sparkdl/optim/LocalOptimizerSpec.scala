/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.dataset.DataSource
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.{RandomGenerator, T}
import org.scalatest.{FlatSpec, Matchers}

object DummyDataSource extends DataSource[(Tensor[Float], Tensor[Float])] {
  var i = 0
  val max = 10
  var isCrossEntropy = true

  def crossEntropy: DataSource[(Tensor[Float], Tensor[Float])] = {
    isCrossEntropy = true
    DummyDataSource
  }

  def mse: DataSource[(Tensor[Float], Tensor[Float])] = {
    isCrossEntropy = false
    DummyDataSource
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

  override def reset(): Unit = {
    i = 0
  }

  override def total(): Long = max

  override def finished(): Boolean = i >= max

  override def shuffle(): Unit = {}

  override def next(): (Tensor[Float], Tensor[Float]) = {
    i += 1
    (feature, if (isCrossEntropy) labelCrossEntropy else labelMSE)
  }

  override def hasNext: Boolean = true
}

object TestDummyDataSource extends DataSource[(Tensor[Float], Tensor[Float])] {
  var i = 0
  val max = 10

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

  override def reset(): Unit = {
    i = 0
  }

  override def total(): Long = max

  override def finished(): Boolean = i >= max

  override def shuffle(): Unit = {}

  override def next(): (Tensor[Float], Tensor[Float]) = {
    i += 1
    (feature, labelCrossEntropy)
  }

  override def hasNext: Boolean = i < max
}

class LocalOptimizerSpec extends FlatSpec with Matchers {
  "Local Optimizer" should "train model well with CrossEntropy and SGD" in {
    RandomGenerator.RNG.setSeed(1000)
    val mlp = new Sequential[Tensor[Float], Tensor[Float], Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)
    val optimizer = new LocalOptimizer[Float](
      DummyDataSource.crossEntropy,
      mlp,
      new ClassNLLCriterion[Float],
      new SGD[Float](),
      T("learningRate" -> 20.0),
      Trigger.maxEpoch(100)
    )

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.max(1)._2.valueAt(1, 1) should be(1.0)
    test.max(1)._2.valueAt(1, 2) should be(2.0)
  }

  it should "train model well with MSE and SGD" in {
    RandomGenerator.RNG.setSeed(1000)
    val mlp = new Sequential[Tensor[Float], Tensor[Float], Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val optimizer = new LocalOptimizer[Float](
      DummyDataSource.mse,
      mlp,
      new MSECriterion[Float],
      new SGD[Float](),
      T("learningRate" -> 20.0),
      Trigger.maxEpoch(10)
    )

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.valueAt(1, 1) < 0.5 should be(true)
    test.valueAt(2, 1) > 0.5 should be(true)
  }

  it should "train model with CrossEntropy and LBFGS" in {
    RandomGenerator.RNG.setSeed(1000)
    val mlp = new Sequential[Tensor[Float], Tensor[Float], Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val optimizer = new LocalOptimizer[Float](
      DummyDataSource.crossEntropy,
      mlp,
      new ClassNLLCriterion[Float],
      new LBFGS[Float](),
      T(),
      Trigger.maxEpoch(100)
    )

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.max(1)._2.valueAt(1, 1) should be(1.0)
    test.max(1)._2.valueAt(1, 2) should be(2.0)
  }

  it should "train model with MSE and LBFGS" in {
    RandomGenerator.RNG.setSeed(1000)
    val mlp = new Sequential[Tensor[Float], Tensor[Float], Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)
    val (weight, grad) = mlp.getParameters()
    weight.fill(0.125f)

    val optimizer = new LocalOptimizer[Float](
      DummyDataSource.mse,
      mlp,
      new MSECriterion[Float],
      new LBFGS[Float](),
      T(),
      Trigger.maxEpoch(100)
    )

    val result = optimizer.optimize()
    val test = result.forward(Tensor[Float](Storage[Float](
      Array[Float](
        0, 1, 0, 1,
        1, 0, 1, 0
      )), storageOffset = 1, size = Array(2, 4)))
    test.valueAt(1, 1) < 0.5 should be(true)
    test.valueAt(2, 1) > 0.5 should be(true)
  }

  it should "get correct validation result" in {
    RandomGenerator.RNG.setSeed(1000)
    val mlp = new Sequential[Tensor[Float], Tensor[Float], Float]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)
    val optimizer = new LocalOptimizer[Float](
      DummyDataSource.crossEntropy,
      TestDummyDataSource,
      mlp,
      new ClassNLLCriterion[Float],
      new SGD[Float](),
      T("learningRate" -> 20.0),
      Trigger.maxEpoch(100)
    )
    optimizer.setValidationTrigger(Trigger.everyEpoch)
    optimizer.addValidation(new Top1Accuracy[Float])
    optimizer.optimize()
  }
}
