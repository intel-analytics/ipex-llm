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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
class SampleSpec extends FlatSpec with Matchers {
  def newMiniBatch[T: ClassTag](
        samples: Array[Sample[T]],
        featurePadding: Option[Array[Tensor[T]]] = None,
        featureFixedLength: Option[Array[Int]] = None,
        featureIncrement: Option[Array[Int]] = None,
        labelPadding: Option[Array[T]] = None,
        labelFixedLength: Option[Array[Int]] = None,
        labelIncrement: Option[Array[Int]] = None)(
        implicit ev: TensorNumeric[T]): MiniBatch[T] = {

    MiniBatch[T](samples(0).numFeature(), samples(0).numLabel(),
      Some(PaddingParam(featurePadding, featureFixedLength, featureIncrement,
      labelPadding, labelFixedLength, labelIncrement))).setValue(samples)
  }

  "SampleSpec with Float Tensor input and Tensor label" should "initialize well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](tensorInput1, tensorLabel1)
    sample.feature should be (tensorInput1)
    sample.label should be (tensorLabel1)
  }

  "TensorSample" should "clone well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](tensorInput1, tensorLabel1)
    val otherSample = sample.clone()
    sample.feature should be (otherSample.feature)
    sample.label should be (otherSample.label)
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
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1), Tensor[Float](1).fill(3))
    val result = newMiniBatch(samples)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)
    val exceptedTarget = Tensor[Float](3, 1).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorSample] toMiniBatch with feature padding" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](1).fill(3))
    val result = newMiniBatch[Float](samples, featurePadding)
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](3).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](1).fill(3))
    val result = newMiniBatch(samples, featurePadding = featurePadding,
      labelPadding = Some(Array(-1f)))
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](3).fill(3))
    val result = newMiniBatch[Float](samples, featurePadding = featurePadding,
      featureFixedLength = Some(Array(4)), labelPadding = Some(Array(-1)),
      labelFixedLength = Some(Array(4)))
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
      1, -1, -1, -1,
      2, 2, -1, -1,
      3, 3, 3, -1
    )), 1, Array(3, 4))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorSample] toMiniBatch with increment" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), Tensor[Float](3).fill(3))
    val result = newMiniBatch[Float](samples, featurePadding = featurePadding,
      featureIncrement = Some(Array(2)), labelPadding = Some(Array(-1)),
      labelIncrement = Some(Array(1)))
    val exceptedInput = Tensor[Float](Storage(Array[Float](
      1, 2, 3,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      10, 11, 12,
      13, 14, 15,
      -1f, -2f, -3f,
      -1f, -2f, -3f,
      -1f, -2f, -3f,

      19, 20, 21,
      22, 23, 24,
      25, 26, 27,
      -1f, -2f, -3f,
      -1f, -2f, -3f
    )), 1, Array(3, 5, 3))
    val exceptedTarget = Tensor[Float](Storage(Array[Float](
      1, -1, -1, -1,
      2, 2, -1, -1,
      3, 3, 3, -1
    )), 1, Array(3, 4))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorTSample] toMiniBatch" should "return right result" in {
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1), 3)
    val result = newMiniBatch(samples)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)
    val exceptedTarget = Tensor[Float](3, 1).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[TensorTSample] toMiniBatch with feature padding" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), 3)
    val result = newMiniBatch[Float](samples, featurePadding)
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

  "Array[TensorTSample] toMiniBatch with fixedlength" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1), 1)
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1), 2)
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1), 3)
    val result = newMiniBatch[Float](
      samples, featurePadding,
      labelPadding = Some(Array(-1)),
      featureFixedLength = Some(Array(4)))
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
    val exceptedTarget = Tensor[Float](Storage(Array[Float](1, 2, 3 )), 1, Array(3, 1))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch" should "return right result" in {
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(7, 12, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](2, 3).range(13, 18, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](1).fill(3))
    val result = newMiniBatch(samples)
    val exceptedInput = T(Tensor[Float](3, 2, 3).range(1, 18, 1), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)
    val exceptedTarget = Tensor[Float](3, 1).range(1, 3, 1)

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch with feature padding" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](1)))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](1).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](1).fill(3))
    val result = newMiniBatch[Float](samples, featurePadding)
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](1)))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](3).fill(3))
    val result = newMiniBatch[Float](samples, featurePadding, labelPadding = Some(Array(-1)))
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](1)))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](3).fill(3))
    val result = newMiniBatch[Float](
      samples, featurePadding,
      Some(Array(4, 3)),
      None,
      Some(Array(-1)),
      Some(Array(4))
      )
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
      1, -1, -1, -1,
      2, 2, -1, -1,
      3, 3, 3, -1
    )), 1, Array(3, 4))

    result.getInput() should be (exceptedInput)
    result.getTarget() should be (exceptedTarget)
  }

  "Array[ArrayTensorSample] toMiniBatch with fixedlength(4, -1)" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](1)))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)), Tensor[Float](2).fill(2))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)), Tensor[Float](3).fill(3))
    val result = newMiniBatch[Float](
      samples, featurePadding,
      Some(Array(4, -1)),
      None,
      Some(Array(-1)),
      Some(Array(-1))
    )
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
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(7, 12, 1))
    samples(2) = Sample[Float](Tensor[Float](2, 3).range(13, 18, 1))
    val result = newMiniBatch(samples)
    val exceptedInput = Tensor[Float](3, 2, 3).range(1, 18, 1)

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabledTensorSample] toMiniBatch with feature padding" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1))
    val result = newMiniBatch[Float](samples, featurePadding = featurePadding)
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Tensor[Float](1, 3).range(1, 3, 1))
    samples(1) = Sample[Float](Tensor[Float](2, 3).range(10, 15, 1))
    samples(2) = Sample[Float](Tensor[Float](3, 3).range(19, 27, 1))
    val result = newMiniBatch[Float](
      samples, featurePadding = featurePadding, featureFixedLength = Some(Array(4)))
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
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(7, 12, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](2, 3).range(13, 18, 1),
      Tensor[Float](3).fill(3)))
    val result = newMiniBatch(samples)
    val exceptedInput = T(Tensor[Float](3, 2, 3).range(1, 18, 1), Tensor[Float](3, 3))
    exceptedInput[Tensor[Float]](2)(1).fill(1)
    exceptedInput[Tensor[Float]](2)(2).fill(2)
    exceptedInput[Tensor[Float]](2)(3).fill(3)

    result.getInput() should be (exceptedInput)
  }

  "Array[UnlabeledArrayTensorSample] toMiniBatch with padding" should "return right result" in {
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](1)))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)))
    val result = newMiniBatch[Float](samples, featurePadding)
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
    val featurePadding = Some(Array(Tensor[Float](Storage(Array(-1f, -2f, -3f))),
      Tensor[Float](Storage(Array(-4f)))))
    val samples: Array[Sample[Float]] = new Array[Sample[Float]](3)
    samples(0) = Sample[Float](Array(Tensor[Float](1, 3).range(1, 3, 1),
      Tensor[Float](3).fill(1)))
    samples(1) = Sample[Float](Array(Tensor[Float](2, 3).range(10, 15, 1),
      Tensor[Float](3).fill(2)))
    samples(2) = Sample[Float](Array(Tensor[Float](3, 3).range(19, 27, 1),
      Tensor[Float](3).fill(3)))
    val result = newMiniBatch[Float](
      samples, featurePadding = featurePadding, featureFixedLength = Some(Array(4, 4)))
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
    )), 1, Array(3, 4, 3)),
      Tensor[Float](Storage(Array[Float](
        1, 1, 1, -4,
        2, 2, 2, -4,
        3, 3, 3, -4
      )), 1, Array(3, 4)))

    result.getInput() should be (exceptedInput)
  }
}
