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

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn, SparkContextLifeCycle}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl._

class EvaluatorSpec extends SparkContextLifeCycle with Matchers {

  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "evaluator"

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "Evaluator" should "be correct" in {
    RNG.setSeed(100)
    val tmp = new Array[Sample[Float]](100)
    var i = 0
    while (i < tmp.length) {
      val input = Tensor[Float](28, 28).fill(0.8f)
      val label = Tensor[Float](1).fill(1.0f)
      tmp(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = DataSet.array(tmp, sc).toDistributed().data(train = false)

    val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.44278f, 100))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }

  "Evaluator MiniBatch" should "be correct" in {
    RNG.setSeed(100)
    val tmp = new Array[MiniBatch[Float]](25)
    var i = 0
    while (i < tmp.length) {
      val input = Tensor[Float](4, 28, 28).fill(0.8f)
      val label = Tensor[Float](4).fill(1.0f)
      tmp(i) = MiniBatch(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = DataSet.array(tmp, sc).toDistributed().data(train = false)

    val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.44278f, 100))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }

  "Evaluator different MiniBatch" should "be correct" in {
    RNG.setSeed(100)
    val tmp = new Array[MiniBatch[Float]](25)
    var i = 1
    while (i <= tmp.length) {
      val input = Tensor[Float](i, 28, 28).fill(0.8f)
      val label = Tensor[Float](i).fill(1.0f)
      tmp(i - 1) = MiniBatch(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = DataSet.array(tmp, sc).toDistributed().data(train = false)

    val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    result(0)._1 should be (new AccuracyResult(0, 325))
    result(1)._1 should be (new AccuracyResult(325, 325))
    result(2)._1 should be (new LossResult(748.93896f, 325))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }
}
