/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.SparkContext
import org.scalatest.{FlatSpec, Matchers}

class PredictorSpec extends FlatSpec with Matchers {
  "prediction" should "be correct" in {
    Engine.init(1, 1, true)
    RNG.setSeed(100)
    val sc = new SparkContext("local[1]", "DistriValidator")
    val data = new Array[Sample[Float]](100)
    var i = 0
    val batchsize = 10
    while (i < data.length) {
      val input = Tensor[Float](28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = sc.parallelize(data, 2)
    dataSet.getNumPartitions should be (2)
    val predictor = Predictor(model, dataSet)
    val result = predictor.predict(batchsize)
    val prob = result.map(_.clone()).collect()
    prob(0).size() should be (Array(10, 10))
    prob(0).valueAt(2, 2) should be (-2.2125769f)
    prob(1).size() should be (Array(10, 10))
    prob(1).valueAt(2, 2) should be (-2.2041807f)
  }
}
