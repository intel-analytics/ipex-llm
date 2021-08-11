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

import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.utils.SparkContextLifeCycle
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class ValidatorSpec extends SparkContextLifeCycle with Matchers {

  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "validator"

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "DistriValidator" should "be correct" in {
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
    val dataSet = DataSet.array(tmp, sc).transform(SampleToMiniBatch(1))
    val validator = Validator(model, dataSet)

    val result = validator.test(Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.4428f, 100))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }

  "LocalValidator" should "be correct" in {
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
    val dataSet = DataSet.array(tmp).transform(SampleToMiniBatch(1))
    val validator = Validator(model, dataSet)

    val result = validator.test(Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.4428f, 100))
  }
}
