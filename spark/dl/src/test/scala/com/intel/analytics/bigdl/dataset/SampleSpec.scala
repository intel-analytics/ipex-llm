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
    val sample = new TensorSample[Float](tensorInput1, tensorLabel1)
    sample.featureTensor should be (tensorInput1)
    sample.labelTensor should be (tensorLabel1)
  }

  "TensorSample" should "clone well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = new TensorSample[Float](tensorInput1, tensorLabel1)
    val otherSample = sample.clone()
    sample.featureTensor should be (otherSample.featureTensor)
    sample.labelTensor should be (otherSample.labelTensor)
  }
}
