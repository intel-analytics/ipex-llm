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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SampleSpec extends FlatSpec with Matchers {
  "SampleSpec with Float Tensor input and Tensor label" should "initialize well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](tensorInput1, tensorLabel1)
    sample.feature() should be (tensorInput1)
    sample.label() should be (tensorLabel1)
  }
  "SampleSpec with Float Tensor input and Tensor label" should "set well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float]()
    sample.set(tensorInput1.storage().array(),
      tensorLabel1.storage().array(),
      tensorInput1.size,
      tensorLabel1.size)
    sample.feature() should be (tensorInput1)
    sample.label() should be (tensorLabel1)
  }
  "SampleSpec with Float Tensor input and Tensor label" should "clone well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](tensorInput1, tensorLabel1)
    val otherSample = sample.clone()
    sample.feature() should be (otherSample.feature())
    sample.label() should be (otherSample.label())
  }
  "SampleSpec with Float Tensor input and Tensor label" should "copyFromFeature" +
    "and copyFromLabel well" in {
    val size = 4
    val input1 = new LabeledBGRImage(size, size)
    val label1 = new LabeledBGRImage(size, size)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(size, size))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(size, size))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](tensorInput1, tensorLabel1)
    val input2 = new Array[Float](size*size)
    val label2 = new Array[Float](size*size)
    sample.copyFromFeature(input2, 0, size*size)
    sample.copyFromLabel(label2, 0, size*size)

    var i = 0
    while (i < input2.length) {
      input2(i) should be (sample.feature().storage().array()(i))
      i += 1
    }
    i = 0
    while (i < label2.length) {
      label2(i) should be (sample.label().storage().array()(i))
      i += 1
    }
  }
}
