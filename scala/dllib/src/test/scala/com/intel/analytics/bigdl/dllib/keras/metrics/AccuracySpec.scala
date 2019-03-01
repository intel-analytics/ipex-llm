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

package com.intel.analytics.zoo.pipeline.api.keras.metrics

import com.intel.analytics.bigdl.optim.AccuracyResult
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class AccuracySpec extends FlatSpec with Matchers {

  "top1 accuracy for categoricalAccuracy" should "be correct on 2d tensor" in {
    val output = Tensor[Double](Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0
    )), 1, Array(3, 4))

    val target = Tensor[Double](Storage(Array[Double](
      0, 0, 0, 1,
      0, 0, 1, 0,
      1, 0, 0, 0
    )), 1, Array(3, 4))
    val validation = new CategoricalAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(2, 3)
    result should be(test)
  }

  "accuracy for binaryAccuracy" should "be correct" in {
    val output = Tensor(Storage(Array[Double](0.3)), 1, Array(1))
    val target = Tensor(Storage(Array[Double](0)))
    val validation = new BinaryAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(1, 1)
    result should be(test)
  }

  "accuracy for SparseCategoricalAccuracy" should "be correct" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0
    )), 1, Array(3, 4))

    val target = Tensor(Storage(Array[Double](
      3,
      1,
      2
    )))

    val validation = new Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(2, 3)
    result should be(test)
  }

  "top1 accuracy using 0-based label" should "be correct on 2d tensor" in {
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
      3,
      1,
      0,
      2,
      1,
      1,
      1,
      3
    )))

    val validation = new Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }

  "accuracy for binary classification" should "ignore zero-based label" in {
    val output = Tensor(Storage(Array[Double](0.3, 0.2, 0.6, 0.8)), 1, Array(4, 1))
    val target = Tensor(Storage(Array[Double](0, 1, 1, 1)))
    val validation = new Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(3, 4)
    result should be(test)
  }

  "accuracy for binary classification" should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](0.3)), 1, Array(1))
    val target = Tensor(Storage(Array[Double](0)))
    val validation = new Accuracy[Double](zeroBasedLabel = false)
    val result = validation(output, target)
    val test = new AccuracyResult(1, 1)
    result should be(test)
  }

  "top1 accuracy using 1-based label" should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1
    )))

    val target1 = Tensor(Storage(Array[Double](
      4
    )))

    val target2 = Tensor(Storage(Array[Double](
      2
    )))

    val validation = new Accuracy[Double](zeroBasedLabel = false)
    val result1 = validation(output, target1)
    val test1 = new AccuracyResult(1, 1)
    result1 should be(test1)

    val result2 = validation(output, target2)
    val test2 = new AccuracyResult(0, 1)
    result2 should be(test2)
  }

  "Top5 accuracy using 0-based label" should "be correct on 2d tensor" in {
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
      3,
      1,
      0,
      2,
      1,
      1,
      1,
      3
    )))

    val validation = new Top5Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }

  "Top5 accuracy using 1-based label" should "be correct on 1d tensor" in {
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

    val validation = new Top5Accuracy[Double](zeroBasedLabel = false)
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
}
