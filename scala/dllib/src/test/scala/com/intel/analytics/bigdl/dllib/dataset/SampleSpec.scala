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
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
class SampleSpec extends FlatSpec with Matchers {
  private def newMiniBatch[T: ClassTag](
        samples: Array[Sample[T]],
        featurePadding: Option[Array[Tensor[T]]] = None,
        featureFixedLength: Option[Array[Int]] = None,
        featureIncrement: Option[Array[Int]] = None,
        labelPadding: Option[Array[T]] = None,
        labelFixedLength: Option[Array[Int]] = None,
        labelIncrement: Option[Array[Int]] = None)(
        implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val featureParam = if (featureFixedLength.isDefined) {
      PaddingParam(featurePadding, FixedLength(featureFixedLength.get))
    } else if (featureIncrement.isDefined) {
      PaddingParam(featurePadding, PaddingLongest(featureIncrement.get))
    } else {
      PaddingParam(featurePadding)
    }

    val newLabelPadding = if (labelPadding.isDefined) {
      Some(labelPadding.get.map(v => Tensor[T](1).fill(v)))
    } else {
      None
    }

    val labelParam = if (labelFixedLength.isDefined) {
      PaddingParam(newLabelPadding, FixedLength(labelFixedLength.get))
    } else if (labelIncrement.isDefined) {
      PaddingParam(newLabelPadding, PaddingLongest(labelIncrement.get))
    } else {
      PaddingParam(newLabelPadding)
    }

    MiniBatch[T](samples(0).numFeature(), samples(0).numLabel(),
      Some(featureParam), Some(labelParam)).set(samples)
  }

  "Get feature and label from single sample" should "work fine" in {
    val feature = Tensor[Float](2, 2).fill(1.0f)
    val label = Tensor[Float](2).fill(1.0f)
    val sample = ArraySample(feature, label)

    val fetchedFeature = sample.feature()

    val fetchedLabel = sample.label()

    fetchedFeature should be (feature)

    fetchedLabel should be (label)
  }

  "Get label from single sample without label" should "work fine" in {
    val feature = Tensor[Float](2, 2).fill(1.0f)
    val sample = ArraySample(feature)
    val fetchedFeature = sample.feature()

    val fetchedLabel = sample.label()

    fetchedFeature should be (feature)

    fetchedLabel should be (null)
  }

  "Get feature and label from multiple samples" should "work fine" in {
    val feature1 = Tensor[Float](2, 2).fill(1.0f)
    val label1 = Tensor[Float](2).fill(1.0f)

    val feature2 = Tensor[Float](2, 2).fill(2.0f)

    val sample = ArraySample(Array(feature1, feature2), Array(label1))

    val fetchedFeature1 = sample.feature(0)

    val fetchedLabel1 = sample.label(0)

    val fetchedFeature2 = sample.feature(1)

    val fetchedLabel2 = sample.label(1)

    fetchedFeature1 should be (feature1)

    fetchedLabel1 should be (label1)

    fetchedFeature2 should be (feature2)

    fetchedLabel2 should be (null)

  }

  "Get feature and label from TensorSample" should "work properly" in {
    val feature1 = Tensor[Float](2, 2).fill(1.0f)
    val label1 = Tensor[Float](2).fill(1.0f)

    val feature2 = Tensor[Float](2, 2).fill(2.0f)

    val sample = TensorSample(Array(feature1, feature2), Array(label1))

    val fetchedFeature1 = sample.feature(0)

    val fetchedLabel1 = sample.label(0)

    val fetchedFeature2 = sample.feature(1)

    val fetchedLabel2 = sample.label(1)

    fetchedFeature1 should be (feature1)

    fetchedLabel1 should be (label1)

    fetchedFeature2 should be (feature2)

    fetchedLabel2 should be (null)

  }

  "create Sample" should "work fine" in {
    val st1 = Tensor.sparse(Tensor.range(1, 10, 1))
    val st2 = Tensor.sparse(Tensor.range(1, 10, 1))
    val dt1 = Tensor.range(1, 10, 1)
    val dt2 = Tensor.range(1, 10, 1)
    val label1 = Tensor(1).fill(1)
    val label2 = Tensor(1).fill(2)

    Sample(st1)
    Sample(dt1)
    Sample(Array(st1, st2))
    Sample(Array(dt1, st2))
    Sample(Array(dt1, dt2))

    Sample(st1, label1)
    Sample(dt1, label1)
    Sample(dt1, 1f)
    Sample(st1, 1f)
    Sample(Array(st1, st2), label1)
    Sample(Array(dt1, st2), label1)
    Sample(Array(dt1, dt2), label1)

    Sample(Array(st1, st2), Array(label1, label2))
    Sample(Array(dt1, st2), Array(label1, label2))
    Sample(Array(dt1, dt2), Array(label1, label2))
  }

  "Hashcode" should "work fine" in {
    val sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    println(sample1.hashCode())

    val sample2 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    println(sample2.hashCode())

    val sample3 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    println(sample3.hashCode())

    val sample4 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    println(sample4.hashCode())

    val sample5 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1f)
    println(sample5.hashCode())
  }

  "equals" should "work fine" in {
    var sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    var sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    sample1.equals(sample2) should be(true)

    sample1 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    sample2 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    sample1.equals(sample2) should be(true)

    sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    sample1.equals(sample2) should be(true)

    sample1 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    sample2 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    sample1.equals(sample2) should be(true)

    sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1f)
    sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1f)
    sample1.equals(sample2) should be(true)
  }

  "equals" should "work fine2" in {
    var sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    var  sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(2))
    sample1.equals(sample2) should be (false)

    sample1 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    sample2 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)), Tensor[Float](1).fill(1))
    sample1.equals(sample2) should be (false)

    sample1 = Sample[Float](Tensor[Float](2, 3).range(2, 7, 1))
    sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1))
    sample1.equals(sample2) should be (false)

    sample1 = Sample[Float](Array(Tensor[Float](3, 2).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    sample2 = Sample[Float](Array(Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](3).fill(1)))
    sample1.equals(sample2) should be (false)

    sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 2f)
    sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), 1f)
    sample1.equals(sample2) should be (false)

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

  "SampleSpec with Float Tensor input and Tensor label" should "set well" in {
    val input1 = new LabeledBGRImage(32, 32)
    val label1 = new LabeledBGRImage(32, 32)
    val tensorInput1 = Tensor[Float](Storage[Float](input1.content), 1, Array(3, 32, 32))
    val tensorLabel1 = Tensor[Float](Storage[Float](label1.content), 1, Array(3, 32, 32))
    tensorInput1.rand()
    tensorLabel1.rand()
    val sample = Sample[Float](Tensor[Float](3, 32, 32), Tensor[Float](3, 32, 32))
    sample.set(tensorInput1.storage().array(),
      tensorLabel1.storage().array(),
      tensorInput1.size,
      tensorLabel1.size)
    sample.feature() should be(tensorInput1)
    sample.label() should be(tensorLabel1)
  }

  "Sample.equals" should "return right result" in {
    val sample1 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    val sample2 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1), Tensor[Float](1).fill(1))
    sample1.equals(sample2) should be (true)

    val sample3 = Sample[Float](Tensor[Float](3, 3).range(1, 9, 1), Tensor[Float](1).fill(10))
    val sample4 = Sample[Float](Tensor[Float](2, 3).range(1, 6, 1),
      Tensor[Float](4).range(7, 10, 1))
    sample3.equals(sample4) should be (false)
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
