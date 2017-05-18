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

import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class PredictorSpec extends FlatSpec with Matchers with BeforeAndAfter{
  var sc: SparkContext = null
  val nodeNumber = 1
  val coreNumber = 1

  before {
    Engine.init(nodeNumber, coreNumber, true)
    val conf = new SparkConf().setMaster("local[1]").setAppName("predictor")
    sc = new SparkContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "model.predict" should "be correct" in {
    RNG.setSeed(100)
    val data = new Array[Sample[Float]](97)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = sc.parallelize(data, 2)
    val result = model.predict(dataSet)

    val prob = result.map(_.toTensor[Float].clone()).collect()
    prob(0) should be (model.forward(data(0).asInstanceOf[TensorSample[Float]].featureTensor))
    prob(11) should be (model.forward(data(11).asInstanceOf[TensorSample[Float]].featureTensor))
    prob(31) should be (model.forward(data(31).asInstanceOf[TensorSample[Float]].featureTensor))
    prob(51) should be (model.forward(data(51).asInstanceOf[TensorSample[Float]].featureTensor))
    prob(71) should be (model.forward(data(71).asInstanceOf[TensorSample[Float]].featureTensor))
    prob(91) should be (model.forward(data(91).asInstanceOf[TensorSample[Float]].featureTensor))
  }

  "model.predictClass" should "be correct" in {
    RNG.setSeed(100)
    val data = new Array[Sample[Float]](97)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = sc.parallelize(data, 2)
    val result = model.predictClass(dataSet)

    val prob = result.collect()
    prob(0) should be
    (model.forward(data(0).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(11) should be
    (model.forward(data(11).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(31) should be
    (model.forward(data(31).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(51) should be
    (model.forward(data(51).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(71) should be
    (model.forward(data(71).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(91) should be
    (model.forward(data(91).asInstanceOf[TensorSample[Float]].featureTensor
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
  }
}
