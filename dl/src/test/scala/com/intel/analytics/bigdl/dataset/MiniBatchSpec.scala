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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class MiniBatchSpec extends FlatSpec with Matchers {
  "TensorMiniBatch" should "initialize well" in {
    val input = Tensor[Float](Array(4, 2, 3)).randn()
    val target = Tensor[Float](Array(4)).randn()
    val miniBatch = TensorMiniBatch[Float](input, target)
    miniBatch.size should be (4)
    val narrowMiniBatch = miniBatch.narrow(1, 2, 2)
    narrowMiniBatch.get should be ((input.narrow(1, 2, 2), target.narrow(1, 2, 2)))
  }


  "TensorMiniBatch" should "narrow well" in {
    val input = Tensor[Float](Array(4, 2, 3)).randn()
    val target = Tensor[Float](Array(4)).randn()
    val miniBatch = TensorMiniBatch[Float](input, target)
    val params = miniBatch.get
    params should be ((input, target))

    val miniBatch2 = new TensorMiniBatch[Float](params._1, params._2)
    miniBatch2.size should be (4)

    val batchBuffer = new Array[TensorMiniBatch[Float]](2)
    for (i <- 1 to 2) {
      batchBuffer(i - 1) = miniBatch.narrow(1, i, 1)
    }
    val (a, b) = batchBuffer(0).get
    val length = batchBuffer(0).size
    a should be (input.narrow(1, 1, 1))
    b should be (target.narrow(1, 1, 1))
    length should be (1)
  }

  "TensorMiniBatch" should "get a tuple" in {
    val input = Tensor[Float](Array(4, 2, 3)).randn()
    val target = Tensor[Float](Array(4)).randn()
    val miniBatch = TensorMiniBatch[Float](input, target)

    val (data, label) = miniBatch.get
    data should be (input)
    label should be (target)
  }

  "TableMiniBatch" should "initialize well" in {
    val input = T(Tensor[Float](Array(4, 2)).randn(), Tensor[Float](Array(3, 2)).randn())
    val target = T(Tensor[Float](Array(4)).randn(), Tensor[Float](Array(3)).randn())
    val miniBatch = TableMiniBatch[Float](input, target, 2)
    miniBatch.size should be(2)
    val subBatch = miniBatch.narrow(1, 1, 1)
    val (a, b) = subBatch.get
    a should be (T(input(1).asInstanceOf[Tensor[Float]]))
    b should be (T(target(1).asInstanceOf[Tensor[Float]]))
    subBatch.size should be(1)
  }

  "TableMiniBatch" should "narrow well" in {
    val input = T(Tensor[Float](Array(4, 2)).randn(), Tensor[Float](Array(3, 2)).randn())
    val target = T(Tensor[Float](Array(4)).randn(), Tensor[Float](Array(3)).randn())
    val miniBatch = TableMiniBatch[Float](input, target, 2)
    val params = miniBatch.get
    params should be ((input, target))

    val miniBatch2 = new TableMiniBatch[Float](params._1, params._2, 2)
    miniBatch2.size should be (2)

    val batchBuffer = new Array[TableMiniBatch[Float]](2)
    for (i <- 1 to 2) {
      batchBuffer(i - 1) = miniBatch.narrow(1, i, 1)
    }
    val (a, b) = batchBuffer(0).get
    val length = batchBuffer(0).size
    a should be (T(input(1).asInstanceOf[Tensor[Float]]))
    b should be (T(target(1).asInstanceOf[Tensor[Float]]))
    length should be (1)
  }

  "TableMiniBatch" should "get a tuple" in {
    val input = T(Tensor[Float](Array(4, 2)).randn(), Tensor[Float](Array(3, 2)).randn())
    val target = T(Tensor[Float](Array(4)).randn(), Tensor[Float](Array(3)).randn())
    val miniBatch = TableMiniBatch[Float](input, target, 2)

    val (data, label) = miniBatch.get
    data should be (input)
    label should be (target)
  }
}
