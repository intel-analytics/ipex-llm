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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class ValidationSpec extends FlatSpec with Matchers {
  "treeNN accuracy" should "be correct on 2d tensor" in {
    val output = Tensor[Double](
      T(
        T(0.0, 0.0, 0.1, 0.0),
        T(3.0, 7.0, 0.0, 1.0),
        T(0.0, 1.0, 0.0, 0.0)))

    val target = Tensor[Double](
      T(3.0))

    val validation = new TreeNNAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(1, 1)
    result should be(test)
  }

  "treeNN accuracy" should "be correct on 3d tensor" in {
    val output = Tensor[Double](
      T(
        T(
          T(0.0, 0.0, 0.1, 0.0),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.1, 0.0, 0.0),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.0, 0.0, 0.1),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.0, 0.0, 1.0),
          T(3.0, 0.0, 8.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0))))

    val target = Tensor[Double](
      T(
        T(3.0, 0.0, 0.1, 1.0),
        T(2.0, 0.0, 0.1, 1.0),
        T(3.0, 7.0, 0.0, 1.0),
        T(4.0, 1.0, 0.0, 0.0)))

    val validation = new TreeNNAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(3, 4)
    result should be(test)
  }

  "top1 accuracy" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val validation = new Top1Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }


  "top1 accuracy" should "be correct on 2d tensor for binary inputs" in {
    val output = Tensor(Storage(Array[Double](
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      0
    )), 1, Array(8, 1))

    val target = Tensor(Storage(Array[Double](
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      1
    )))

    val validation = new Top1Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(3, 8)
    result should be(test)
  }

  it should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1
    )))

    val target1 = Tensor(Storage(Array[Double](
      4
    )))

    val target2 = Tensor(Storage(Array[Double](
      2
    )))

    val validation = new Top1Accuracy[Double]()
    val result1 = validation(output, target1)
    val test1 = new AccuracyResult(1, 1)
    result1 should be(test1)

    val result2 = validation(output, target2)
    val test2 = new AccuracyResult(0, 1)
    result2 should be(test2)
  }

  "Top5 accuracy" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 8, 1, 2, 0, 0, 0,
      0, 1, 0, 0, 2, 3, 4, 6,
      1, 0, 0, 0.6, 0.1, 0.2, 0.3, 0.4,
      0, 0, 1, 0, 0.5, 1.5, 2, 0,
      1, 0, 0, 6, 2, 3, 4, 5,
      0, 0, 1, 0, 1, 1, 1, 1,
      0, 0, 0, 1, 1, 2, 3, 4,
      0, 1, 0, 0, 2, 4, 3, 2
    )), 1, Array(8, 8))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val validation = new Top5Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }

  it should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0.1, 0.2, 0.6, 0.01, 0.005, 0.005, 0.05, 0.03
    )))

    val target1 = Tensor(Storage(Array[Double](
      2
    )))

    val target2 = Tensor(Storage(Array[Double](
      5
    )))

    val target3 = Tensor(Storage(Array[Double](
      3
    )))

    val target4 = Tensor(Storage(Array[Double](
      7
    )))

    val validation = new Top5Accuracy[Double]()
    val result1 = validation(output, target1)
    val test1 = new AccuracyResult(1, 1)
    result1 should be(test1)

    val result2 = validation(output, target2)
    val test2 = new AccuracyResult(0, 1)
    result2 should be(test2)

    val result3 = validation(output, target3)
    val test3 = new AccuracyResult(1, 1)
    result3 should be(test3)

    val result4 = validation(output, target4)
    val test4 = new AccuracyResult(1, 1)
    result4 should be(test4)
  }

  "MAE" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0.1, 0.15, 0.7, 0.1, 0.05,
      0.1, 0.6, 0.1, 0.1, 0.1,
      0.8, 0.05, 0.05, 0.05, 0.05,
      0.1, 0.05, 0.7, 0.1, 0.05
    )), 1, Array(4, 5))

    val target = Tensor(Storage(Array[Double](
      4,
      3,
      4,
      2
    )))

    val validation = new MAE[Double]()
    val result = validation(output, target)
    val test = new LossResult(1.5f, 1)
    result should be(test)
  }
}
