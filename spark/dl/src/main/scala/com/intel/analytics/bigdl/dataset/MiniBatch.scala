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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag


/**
 * A interface for MiniBatch.
 * A MiniBatch contains a few samples.
 *
 * @tparam T Numeric type
 */
trait MiniBatch[T] extends Serializable{
  /**
   * Get the number of samples in this MiniBatch
   * @return size How many samples in this MiniBatch
   */
  def size(): Int

  /**
   * Slice this MiniBatch to a smaller MiniBatch with offset and length
   * @param offset offset, counted from 1
   * @param length length
   * @return A smaller MiniBatch
   */
  def slice(offset: Int, length: Int): MiniBatch[T]

  /**
   * Get input in this MiniBatch.
   * @return input Activity
   */
  def getInput(): Activity

  /**
   * Get target in this MiniBatch
   * @return target Activity
   */
  def getTarget(): Activity

  @deprecated("Old interface", "0.2.0")
  def data(): Tensor[T] = {
    require(this.isInstanceOf[TensorMiniBatch[T]], "Deprecated method," +
      " Only support TensorMiniBatch.")
    this.asInstanceOf[TensorMiniBatch[T]].input
  }

  @deprecated("Old interface", "0.2.0")
  def labels(): Tensor[T] = {
    require(this.isInstanceOf[TensorMiniBatch[T]], "Deprecated method," +
      " Only support TensorMiniBatch.")
    this.asInstanceOf[TensorMiniBatch[T]].input
  }

  def setValue(samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): this.type
}

/**
 * A MiniBatch with [[Tensor]] input and [[Tensor]] target.
 * The size of first dimension in input and target should be the mini-batch size.
 *
 * @param input input Tensor
 * @param target target Tensor
 * @tparam T Numeric type
 */
class TensorMiniBatch[T: ClassTag](
      val input: Tensor[T],
      val target: Tensor[T]) extends MiniBatch[T]{
  require(input.size(1) == target.size(1))

  override def size(): Int = {
    input.size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    MiniBatch(input.narrow(1, offset, length), target.narrow(1, offset, length))
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }

  override def setValue(samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    // todo
    throw new UnsupportedOperationException("unimplemented")
  }

}

/**
 * A MiniBatch with [[com.intel.analytics.bigdl.utils.Table]] input and [[Tensor]] target.
 * The size of first dimension in input's first tensor and target is the mini-batch size.
 *
 * @param inputData input Table
 * @param targetData target Tensor
 * @tparam T Numeric type
 * @since 0.2.0
 */
class ArrayTensorMiniBatch[T: ClassTag](
    val inputData: Array[Tensor[T]],
    val targetData: Array[Tensor[T]],
    featurePadding: Option[Array[Tensor[T]]] = None,
    featureFixedLength: Option[Array[Int]] = None,
    featureIncrement: Option[Array[Int]] = None,
    labelPadding: Option[Array[T]] = None,
    labelFixedLength: Option[Array[Int]] = None,
    labelIncrement: Option[Array[Int]] = None) extends MiniBatch[T]{
  require(inputData.length > 0, "Input data in MiniBatch is empty.")
  private lazy val input: Activity = if (inputData.length == 1) {
    inputData(0)
  } else {
    T.array(inputData.map(_.asInstanceOf[Any]))
  }

  private lazy val target: Activity = if (targetData.length == 0) {
    null
  } else if (targetData.length == 1) {
    targetData(0)
  } else {
    T.array(targetData.map(_.asInstanceOf[Any]))
  }

  override def size(): Int = {
    inputData(0).size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val is = new Array[Tensor[T]](inputData.length)
    val ts = new Array[Tensor[T]](targetData.length)
    var b = 0
    while(b < inputData.size) {
      is(b) = inputData(b).narrow(1, offset, length)
      b += 1
    }
    b = 0
    while(b < targetData.size) {
      ts(b) = targetData(b).narrow(1, offset, length)
      b += 1
    }

    MiniBatch(is, ts)
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }

  def setValue(samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0)
    require(samples(0).isInstanceOf[ArraySample[T]],
      "ArrayTensorMiniBatch: Only support ArraySample")
    val s = samples.map(_.asInstanceOf[ArraySample[T]])

    val longestFeature = MiniBatch.findLongestFeatures(samples)
    val longestLabel = MiniBatch.findLongestLabels(samples)

    MiniBatch.arraySampleToMiniBatch(s, this, longestFeature, longestLabel, featurePadding,
      featureFixedLength, featureIncrement, labelPadding, labelFixedLength, labelIncrement)
    this
  }
}

object MiniBatch {
  def apply[T: ClassTag](
    nInputs: Int,
    nTargets: Int,
    featurePadding: Option[Array[Tensor[T]]],
    featureFixedLength: Option[Array[Int]],
    featureIncrement: Option[Array[Int]],
    labelPadding: Option[Array[T]],
    labelFixedLength: Option[Array[Int]],
    labelIncrement: Option[Array[Int]])(
    implicit ev: TensorNumeric[T]): MiniBatch[T] = {

    new ArrayTensorMiniBatch[T](Array.tabulate(nInputs)(_ => Tensor[T]()),
      Array.tabulate(nTargets)(_ => Tensor[T]()),
      featurePadding, featureFixedLength, featureIncrement,
      labelPadding, labelFixedLength, labelIncrement)
  }

  def apply[T: ClassTag](input: Tensor[T], target: Tensor[T]): MiniBatch[T] = {
    MiniBatch[T](Array(input), Array(target))
  }

  def apply[T: ClassTag](input: Array[Tensor[T]], target: Tensor[T]): MiniBatch[T] = {
    MiniBatch[T](input, Array(target))
  }

  def apply[T: ClassTag](input: Array[Tensor[T]], target: Array[Tensor[T]]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](input, target)
  }

  def apply[T: ClassTag](input: Tensor[T]): MiniBatch[T] = {
    MiniBatch[T](Array(input), new Array[Tensor[T]](0))
  }

  def apply[T: ClassTag](input: Array[Tensor[T]]): MiniBatch[T] = {
    MiniBatch[T](input, new Array[Tensor[T]](0))
  }

  private def resizeData[T: ClassTag](
        data: Array[Tensor[T]],
        sampleSize: Array[Array[Array[Int]]],
        longestData: Array[Int],
        dataFixedLength: Option[Array[Int]] = None,
        dataIncrement: Option[Array[Int]] = None): Unit = {
    val inputSize = new Array[Array[Int]](longestData.length)
    var ii = 0
    while (ii < inputSize.length) {
      inputSize(ii) = Array(sampleSize.length) ++ sampleSize(longestData(ii))(ii)
      ii += 1
    }

    var i = 0
    while (i < inputSize.length) {
      if (dataFixedLength.isDefined) {
        val fixedLength = dataFixedLength.get(i)
        require(fixedLength >= inputSize(i)(1),
          s"${i}th FixedLength=${fixedLength} is smaller than its FeatureLength=${inputSize(i)(1)}")
        inputSize(i)(1) = fixedLength
      } else if (dataIncrement.isDefined) {
        val increment = dataIncrement.get(i)
        inputSize(i)(1) += increment
      }
      data(i).resize(inputSize(i))
      i += 1
    }

  }

  def arraySampleToMiniBatch[T: ClassTag](
      samples: Array[ArraySample[T]],
      miniBatch: ArrayTensorMiniBatch[T],
      longestFeature: Array[Int],
      longestLabel: Array[Int],
      featurePadding: Option[Array[Tensor[T]]] = None,
      featureFixedLength: Option[Array[Int]] = None,
      featureIncrement: Option[Array[Int]] = None,
      labelPadding: Option[Array[T]] = None,
      labelFixedLength: Option[Array[Int]] = None,
      labelIncrement: Option[Array[Int]] = None
      )(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = miniBatch.inputData
    val target = miniBatch.targetData

    val featureSizes = samples.map(_.getFeatureSize())
    val labelSizes = samples.map(_.getLabelSize())
    val unlabeled = if (labelSizes.flatMap(_.flatMap(_.toIterator)).product == 0) true else false
    resizeData(inputs, featureSizes,
      longestFeature, featureFixedLength, featureIncrement)
    if (!unlabeled) {
      resizeData(target, labelSizes,
        longestLabel, labelFixedLength, labelIncrement)
    }

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      var i = 0
      while (i < inputs.length) {
        require(featurePadding.get.length == inputs.length, s"Number of tensor padding should " +
          s"equals to Number of feature tensor in Sample. Excepted ${inputs.length}," +
          s" but got ${featurePadding.get.length}")
        if (inputs(i).dim() == 2) {
          require(featurePadding.get(i).nElement() == 1, s"${i}thFeature is 1D, featurePadding " +
            s"should have only one element, but got ${featurePadding.get(i)}")
        } else {
          require(featurePadding.get(i).dim() == inputs(i).dim() - 2,
            s"${i}thFeature's featurePadding should have the " +
            s"same dimension with the feature in sample. Excepted: ${inputs(i).dim() - 2}, " +
            s"but got ${featurePadding.get(i).dim()}")
        }
        require(featurePadding.get(i).isContiguous(), "featurePadding should be contiguous")
        i += 1
      }
    }

    // init input and target
    if (labelPadding.isDefined) {
      // fill target with labelPadding first.
      var l = 0
      while(l < target.length) {
        target(l).fill(labelPadding.get(l))
        l += 1
      }
    } else {
      target.foreach(_.zero())
    }
    if (!featurePadding.isDefined) {
      inputs.foreach(_.zero())
    }

    // Copy sample data to miniBatch
    var s = 0
    while (s < samples.length) {
      var f = 0
      var offset = 0
      val featureSize = featureSizes(s)
      while (f < inputs.length) {
        val length = featureSize(f).product
        if (featurePadding.isDefined) {
          // copy data
          copy(samples(s).getData(), offset,
            length, inputs(f)(s + 1), featurePadding.get(f))
        } else {
          // copy data without padding.
          copy(samples(s).getData(), offset,
            length, inputs(f)(s + 1))
        }
        f += 1
        offset += length
      }

      if (!unlabeled) {
        var l = 0
        val labelSize = labelSizes(s)
        while (l < target.length) {
          val length = labelSize(l).product
          copy(samples(s).getData(), offset,
            length, target(l)(s + 1))
          l += 1
          offset += length
        }
      }

      s += 1
    }

    miniBatch
  }

  /**
   * Find Sample in Array[Sample] who has the biggest featureLength
   */
  def findLongestFeatures[T: ClassTag](
        samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): Array[Int] = {
    val featureIndices =
      new Array[Int](samples(0).featureLength().length).map(_ => 0)
    var i = 1
    while (i < samples.length) {
      var j = 0
      while (j < featureIndices.length) {
        if (samples(i).featureLength()(j) > samples(featureIndices(j)).featureLength()(j)) {
          featureIndices(j) = i
        }
        j += 1
      }
      i += 1
    }
    featureIndices
  }

  /**
   * Find Sample in Array[Sample] who has the biggest labelLength
   */
  def findLongestLabels[T: ClassTag](
        samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): Array[Int] = {
    val labelIndices =
      new Array[Int](samples(0).labelLength().length).map(_ => 0)
    var i = 1
    while (i < samples.length) {
      var j = 0
      while (j < labelIndices.length) {
        if (samples(i).labelLength()(j) > samples(labelIndices(j)).labelLength()(j)) {
          labelIndices(j) = i
        }
        j += 1
      }
      i += 1
    }
    labelIndices
  }

  /**
   * Copy tensor src to tensor dest with a padding tensor.
   */
  private def copy[T: ClassTag](
      src: Array[T],
      offset: Int,
      length: Int,
      dest: Tensor[T],
      paddingTensor: Tensor[T] = null)(implicit ev: TensorNumeric[T]): Unit = {
    arrayCopy(src,
      offset,
      dest.storage().array(),
      dest.storageOffset() - 1,
      length)
    if (null != paddingTensor) {
      var j = length
      while (j < dest.nElement()) {
        arrayCopy(paddingTensor.storage().array(), paddingTensor.storageOffset() - 1,
          dest.storage().array(), dest.storageOffset() - 1 + j, paddingTensor.nElement())
        j += paddingTensor.nElement()
      }
    }
  }

  /**
   * A wrapper for System.arraycopy
   */
  private def arrayCopy[T: ClassTag](
      src: AnyRef,
      srcPos: Int,
      dest: AnyRef,
      destPos: Int,
      length: Int)(implicit ev: TensorNumeric[T]): Unit = {
    ev.getType() match {
      case DoubleType => Array.copy(src
        .asInstanceOf[Array[Double]],
        srcPos, dest
          .asInstanceOf[Array[Double]], destPos, length)
      case FloatType => System.arraycopy(src
        .asInstanceOf[Array[Float]],
        srcPos, dest
          .asInstanceOf[Array[Float]], destPos, length)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }
}

