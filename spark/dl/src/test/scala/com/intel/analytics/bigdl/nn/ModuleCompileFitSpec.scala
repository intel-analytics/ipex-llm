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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim.{Adam, SGD, Top1Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ModuleCompileFitSpec extends FlatSpec with Matchers with BeforeAndAfter{
  private var sc: SparkContext = _
  private val nodeNumber = 1
  private val coreNumber = 4

  before {
    Engine.setNodeAndCore(nodeNumber, coreNumber)
    sc = new SparkContext(s"local[$coreNumber]", "ModuleCompileFitSpec")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "model compile and fit" should "work properly" in {
    val data = sc.range(0, 8, 1).map { _ =>
      val featureTensor = Tensor[Float](2)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }

    val model = Sequential[Float]()
      .add(Linear(2, 3))
      .add(Sigmoid())
    model.compile(optimizer = new SGD[Float](), loss = MSECriterion[Float]())
    model.fit(data, batchSize = 8)
  }

  "model compile multiple times" should "use the last compile" in {
    val data = sc.range(0, 8, 1).map { _ =>
      val featureTensor = Tensor[Float](2)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }
    val model = Sequential[Float]()
      .add(Linear(2, 3))
    model.compile(optimizer = new SGD[Float](), loss = ClassNLLCriterion[Float]())
    model.compile(optimizer = new Adam[Float](), loss = MSECriterion[Float]())
    model.fit(data, batchSize = 8)
  }

  "model compile and fit with validation" should "work properly" in {
    val data = sc.range(0, 16, 1).map { _ =>
      val featureTensor = Tensor[Float](2)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }
    val validate = sc.range(0, 8, 1).map { _ =>
      val featureTensor = Tensor[Float](2)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](1)
      labelTensor(Array(1)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }
    val model = Sequential[Float]()
      .add(Linear(2, 3))
      .add(Sigmoid())
    model.compile(optimizer = new SGD[Float](), loss = MSECriterion[Float](),
      metrics = Array(new Top1Accuracy[Float]()))
    model.fit(data, batchSize = 8, validationData = validate)
  }

}