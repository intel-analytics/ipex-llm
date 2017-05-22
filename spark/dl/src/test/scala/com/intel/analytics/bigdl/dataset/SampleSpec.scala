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
import com.intel.analytics.bigdl.utils.T
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
    sample.feature() should be(tensorInput1)
    sample.label() should be(tensorLabel1)
  }

  "Array[TensorSample] toMiniBatch" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1), Tensor[Float](1).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch(samples, buf1, buf2)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)
    val exceptedTarget = Tensor[Float](3, 1).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorSample] toMiniBatch with feature padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](1).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2, featurePadding)
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](1, 2, 3)), 1, Array(3, 1))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorSample] toMiniBatch with padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](3).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](1).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2,
      featurePadding, Some(-1))
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](
      1, 1, 1,
      2, 2, -1,
      3, -1, -1
    )), 1, Array(3, 3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorSample] toMiniBatch with fixedlength" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](3).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](
      samples, buf1, buf2, featurePadding, Some(-1), Some(4))
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f
    )), 1, Array(3, 4, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](
      1, -1, -1,
      2, 2, -1,
      3, 3, 3
    )), 1, Array(3, 3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorTSample] toMiniBatch" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1), 3)
    val result = SampleToMiniBatch.samplesToMiniBatch(samples, buf1, buf2)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)
    val exceptedTarget = Tensor[Float](3).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorTSample] toMiniBatch with feature padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), 3)
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2, featurePadding)
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](1, 2, 3)), 1, Array(3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorTSample] toMiniBatch with fixedlength" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), 3)
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](
      samples, buf1, buf2, featurePadding, Some(-1), Some(4))
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f
    )), 1, Array(3, 4, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](1, 2, 3 )), 1, Array(3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(7, 12, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](2, 3).range(13, 18, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](1).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch(samples, buf1, buf2)
    val exceptedInput = T(Tensor[Float](3, 2, 3).range(1, 18, 1), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)
    val exceptedTarget = Tensor[Float](3, 1).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch with feature padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](1).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2, featurePadding)
    val exceptedInput = T(Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3)), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)
    val exceptedTarget = Tensor[Float](Storage(Array[Float](1, 2, 3)), 1, Array(3, 1))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch with padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](3).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2,
      featurePadding, Some(-1))
    val exceptedInput = T(Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3)), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)
    val exceptedTarget = Tensor[Float](Storage(Array[Float](
      1, -1, -1,
      2, 2, -1,
      3, 3, 3
    )), 1, Array(3, 3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch with fixedlength" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](3).fill(3))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](
      samples, buf1, buf2, featurePadding, Some(-1), Some(4))
    val exceptedInput = T(Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f
    )), 1, Array(3, 4, 3)), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)
    val exceptedTarget = Tensor[Float](Storage(Array[Float](
      1, -1, -1,
      2, 2, -1,
      3, 3, 3
    )), 1, Array(3, 3))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[UnlabledTensorSample] toMiniBatch" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1))
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1))
    val result = SampleToMiniBatch.samplesToMiniBatch(samples, buf1)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabledTensorSample] toMiniBatch with feature padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1,
      featurePadding = featurePadding)
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3))

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabledTensorSample] toMiniBatch with fixedlength" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](
      samples, buf1,
      featurePadding = featurePadding,
      fixedLength = Some(4))
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f
    )), 1, Array(3, 4, 3))

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabeledArrayTensorSample] toMiniBatch" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(7, 12, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](2, 3).range(13, 18, 1),
      Tensor[Float](3).fill(3)))
    val result = SampleToMiniBatch.samplesToMiniBatch(samples, buf1)
    val exceptedInput = T(Tensor[Float](3, 2, 3).range(1, 18, 1), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabeledArrayTensorSample] toMiniBatch with padding" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    lazy val buf2 = Array(Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](samples, buf1, buf2, featurePadding)
    val exceptedInput = T(Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27
    )), 1, Array(3, 3, 3)), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabeledArrayTensorSample] toMiniBatch with fixedlength" should "return right result" in {
    lazy val buf1 = Array(Tensor[Float](), Tensor[Float]())
    val featurePadding = Some(Tensor[Float](Storage(Array(-1f, -2f, -3f))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)))
    val result = SampleToMiniBatch.samplesToMiniBatch[Float](
      samples, buf1, featurePadding = featurePadding, fixedLength = Some(4))
    val exceptedInput = T(Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f
    )), 1, Array(3, 4, 3)), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)

    result.getInput() should be (exceptedInput)
  }
}
