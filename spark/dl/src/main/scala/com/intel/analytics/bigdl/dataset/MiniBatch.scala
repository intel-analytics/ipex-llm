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
   * Set a batchSize of this miniBatch.
   * @param batchSize batch Size
   * @return this
   */
  def setBatchSize(batchSize: Int): this.type

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

  /**
   * An deprecated function for single-input/single-target MiniBatch.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   */
  @deprecated("Old interface, use getInput instead", "0.2.0")
  def data(): Tensor[T] = {
    throw new UnsupportedOperationException("MiniBatch.data(): unimplemented deprecated method")
  }

  /**
   * An deprecated function for single-input/single-target MiniBatch.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   */
  @deprecated("Old interface, use getTarget instead", "0.2.0")
  def labels(): Tensor[T] = {
    throw new UnsupportedOperationException("MiniBatch.labels(): unimplemented deprecated method")
  }

  /**
   * Replace the original content of the miniBatch with a set of Sample.
   * @param samples a set of Sample
   * @return self
   */
  def setValue(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type
}

/**
 * Default type of MiniBatch in BigDL.
 * This MiniBatch support both single/multi inputs and single/multi targets.
 * `inputData` store the input tensors, if `inputData.length == 1`, `getInput()` will return
 * a tensor; If `inputData.length > 1`, `getInput()` will return a table.
 * `targetData` store the target tensors, if `targetData.length == 1`, `getTarget()` will return
 * a tensor; If `targetData.length > 1`, `getTarget()` will return a table.
 *
 * @param inputData a set of input tensor
 * @param targetData a set of target tensor
 * @param featurePaddingParam feature padding strategy, see
 *                       [[com.intel.analytics.bigdl.dataset.FeaturePaddingParam]] for details.
 * @param labelPaddingParam   label padding strategy, see
 *                       [[com.intel.analytics.bigdl.dataset.LabelPaddingParam]] for details.
 * @tparam T Numeric type
 * @since 0.2.0
 */
private[bigdl] class ArrayTensorMiniBatch[T: ClassTag](
      val inputData: Array[Tensor[T]],
      val targetData: Array[Tensor[T]],
      featurePaddingParam: Option[FeaturePaddingParam[T]] = None,
      labelPaddingParam: Option[LabelPaddingParam[T]] = None) extends MiniBatch[T]{
  require(inputData.length > 0, "Input data in MiniBatch is empty.")
  private var batchSize = 0

  def setBatchSize(batchSize: Int): this.type = {
    require(batchSize > 0, s"Illegal batch size ${batchSize}")
    this.batchSize = batchSize
    this
  }

  val (featurePadding, featurePaddingStrategy) = if (featurePaddingParam.isDefined) {
    (featurePaddingParam.get.paddingTensor, featurePaddingParam.get.paddingStrategy)
  } else {
    (None, None)
  }

  val (labelPadding, labelPaddingStrategy) = if (labelPaddingParam.isDefined) {
    (labelPaddingParam.get.paddingValue, labelPaddingParam.get.paddingStrategy)
  } else {
    (None, None)
  }


  private val input: Activity = if (inputData.length == 1) {
    inputData(0)
  } else {
    T.array(inputData.map(_.asInstanceOf[Any]))
  }

  private val target: Activity = if (targetData.length == 0) {
    null
  } else if (targetData.length == 1) {
    targetData(0)
  } else {
    T.array(targetData.map(_.asInstanceOf[Any]))
  }

  override def size(): Int = {
    if (inputData(0).nElement() == 0) {
      0
    } else {
      inputData(0).size(1)
    }
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val inputs = new Array[Tensor[T]](inputData.length)
    val targets = new Array[Tensor[T]](targetData.length)
    var b = 0
    while(b < inputData.size) {
      inputs(b) = inputData(b).narrow(1, offset, length)
      b += 1
    }
    b = 0
    while(b < targetData.size) {
      targets(b) = targetData(b).narrow(1, offset, length)
      b += 1
    }

    MiniBatch(inputs, targets)
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }

  override def setValue(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0, "samples is empty")
    require(batchSize == 0 || samples.length <= batchSize, "setValue: samples's size doesn't " +
      s"match mini batch size, excepted ${size()} got ${samples.length}")

    if (featurePaddingParam.isDefined || labelPaddingParam.isDefined) {
      val longestFeature = MiniBatch.findLongestFeatures(samples)
      val longestLabel = MiniBatch.findLongestLabels(samples)

      MiniBatch.arraySampleToMiniBatch[T](samples, this,
        Some(longestFeature), Some(longestLabel),
        featurePadding, featurePaddingStrategy, labelPadding, labelPaddingStrategy)
    } else {
      MiniBatch.arraySampleToMiniBatch[T](samples, this)
    }
    this
  }

  @deprecated("Old interface", "0.2.0")
  override def data(): Tensor[T] = {
    require(targetData.length == 1, "Deprecated method," +
      " Only support TensorMiniBatch.")
    input.asInstanceOf[Tensor[T]]
  }

  @deprecated("Old interface", "0.2.0")
  override def labels(): Tensor[T] = {
    require(inputData.length == 1, "Deprecated method," +
      " Only support TensorMiniBatch.")
    target.asInstanceOf[Tensor[T]]
  }
}

object MiniBatch {
  /**
   * MiniBatch factory method
   * @param nInputs number of inputs
   * @param nTargets number of targets
   * @return
   */
  def apply[T: ClassTag](
        nInputs: Int,
        nTargets: Int,
        featurePaddingParam: Option[FeaturePaddingParam[T]] = None,
        labelPaddingParam: Option[LabelPaddingParam[T]] = None)(
    implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](Array.tabulate(nInputs)(_ => Tensor[T]()),
      Array.tabulate(nTargets)(_ => Tensor[T]()),
      featurePaddingParam, labelPaddingParam)
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
        // 1st Seq is batchSize, 2nd Array is number of features, 3th Array is feature size
        sampleSize: Seq[Array[Array[Int]]],
        longestData: Option[Array[Int]],
        paddingStrategy: Option[PaddingStrategy] = None): Unit = {
    // Size of input data. 1st Array is number of input, 2nd Array is input size.
    val finalSizes = if (longestData.isDefined) {
      val longest = longestData.get
      val sizes = new Array[Array[Int]](longest.length)

      var i = 0
      while (i < sizes.length) {
        // Set i-th input's size
        sizes(i) = Array(sampleSize.length) ++ sampleSize(longest(i))(i)
        i += 1
      }

      if (paddingStrategy.isDefined) {
        paddingStrategy.get.padding(sizes)
      }
      sizes
    } else {
      val sizes = new Array[Array[Int]](sampleSize(0).length)
      var i = 0
      while (i < sizes.length) {
        // Set i-th input's size
        sizes(i) = Array(sampleSize.length) ++ sampleSize(0)(i)
        i += 1
      }
      sizes

    }

    // resize
    var i = 0
    while (i < finalSizes.length) {
      data(i).resize(finalSizes(i))
      i += 1
    }

  }

  private[bigdl] def arraySampleToMiniBatch[T: ClassTag](
      samples: Seq[Sample[T]],
      miniBatch: ArrayTensorMiniBatch[T],
      longestFeature: Option[Array[Int]] = None,
      longestLabel: Option[Array[Int]] = None,
      featurePadding: Option[Array[Tensor[T]]] = None,
      featurePaddingStrategy: Option[PaddingStrategy] = None,
      labelPadding: Option[Array[T]] = None,
      labelPaddingStrategy: Option[PaddingStrategy] = None
      )(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = miniBatch.inputData
    val targets = miniBatch.targetData

    val featureSizes = samples.map(_.getFeatureSize())
    val labelSizes = samples.map(_.getLabelSize())
    val unlabeled = if (labelSizes.flatMap(_.flatMap(_.toIterator)).product == 0) true else false
    resizeData(inputs, featureSizes,
      longestFeature, featurePaddingStrategy)
    if (!unlabeled) {
      resizeData(targets, labelSizes,
        longestLabel, labelPaddingStrategy)
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
      while(l < targets.length) {
        targets(l).fill(labelPadding.get(l))
        l += 1
      }
    } else {
      targets.foreach(_.zero())
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
        while (l < targets.length) {
          val length = labelSize(l).product
          copy(samples(s).getData(), offset,
            length, targets(l)(s + 1))
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
  private[bigdl] def findLongestFeatures[T: ClassTag](
        samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): Array[Int] = {
    val featureIndices =
      new Array[Int](samples(0).numFeature())
    var i = 1
    while (i < samples.length) {
      var j = 0
      while (j < featureIndices.length) {
        if (samples(i).featureLength(j) > samples(featureIndices(j)).featureLength(j)) {
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
  private[bigdl] def findLongestLabels[T: ClassTag](
        samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): Array[Int] = {
    val labelIndices =
      new Array[Int](samples(0).numLabel())
    var i = 1
    while (i < samples.length) {
      var j = 0
      while (j < labelIndices.length) {
        if (samples(i).labelLength(j) > samples(labelIndices(j)).labelLength(j)) {
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
    ev.arraycopy(src,
      offset,
      dest.storage().array(),
      dest.storageOffset() - 1,
      length)
    if (null != paddingTensor) {
      var j = length
      while (j < dest.nElement()) {
        ev.arraycopy(paddingTensor.storage().array(), paddingTensor.storageOffset() - 1,
          dest.storage().array(), dest.storageOffset() - 1 + j, paddingTensor.nElement())
        j += paddingTensor.nElement()
      }
    }
  }
}

/**
 * Feature Padding param for MiniBatch.
 *
 * For constructing a mini batch, we need to make sure all samples' feature and label
 * in this mini batch have the same size. If the size is different, we will pad them
 * to the same size.
 *
 * By default, we will pad the first dimension to the longest size with zero in the MiniBatch.
 * If you want to specify the padding values, you can set `paddingTensor`; If you want to specify
 * the padding length, you can use `PaddingLongest` or `FixedLength`.
 *
 * For example, your feature size is n*m*k,
 *                       you should provide a 2D tensor in a size of m*k.
 *                       If your feature is 1D, you can provide a one-element 1D tensor.
 *
 * For example, we have 3 Sample, and convert them into a MiniBatch.
 * Sample1's feature is a 2*3 tensor {1, 2, 3,
 *                                    4, 5, 6}
 *
 * Sample2's feature is a 1*3 tensor {7, 8, 9}
 *
 * Sample3's feature is a 3*3 tensor {10, 11, 12,
 *                                    13, 14, 15,
 *                                    16, 17, 18}
 *
 * And the paddingTensor is {-1, -2, -3}, use `FixedLength(Array(4))`, the MiniBatch will be
 * a tensor of 3*4*3:
 * {1, 2, 3,
 *  4, 5, 6,
 *  -1, -2, -3,
 *  -1, -2, -3
 *
 *  7, 8, 9,
 *  -1, -2, -3,
 *  -1, -2, -3,
 *  -1, -2, -3
 *
 *  10, 11, 12,
 *  13, 14, 15,
 *  16, 17, 18
 *  -1, -2, -3}
 *
 * @param paddingTensor paddings tensor for the first dimension(by default None,
 *                       meaning zero padding).
 * @param paddingStrategy See [[PaddingLongest]], [[FixedLength]]
 * @tparam T numeric type
 */
case class FeaturePaddingParam[T: ClassTag](
      paddingTensor: Option[Array[Tensor[T]]] = None,
      paddingStrategy: Option[PaddingStrategy] = None)

/**
 * Label Padding param for MiniBatch.
 *
 * For constructing a mini batch, we need to make sure all samples' feature and label
 * in this mini batch have the same size. If the size is different, we will pad them
 * to the same size.
 *
 * By default, we will pad the first dimension to the longest size with zero in the MiniBatch.
 * If you want to specify the padding values, you can set `paddingValue`; If you want to specify
 * the padding length, you can use `PaddingLongest` or `FixedLength`.
 *
 * For example, we have 3 Sample, and convert them into a MiniBatch.
 * Sample1's label is a tensor {1}
 *
 * Sample2's label is a tensor {2, 3}
 *
 * Sample3's feature is a tensor {4, 5, 6}
 *
 * And the paddingValue is `0`, fixedLength is 4, the MiniBatch will be
 * a tensor of 3*4:
 * {1, 0, 0, 0
 *  2, 3, 0, 0
 *  4, 5, 6, 0}
 *
 * @param paddingValue padding value
 * @param paddingStrategy padding strategy
 */
case class LabelPaddingParam[T: ClassTag](
      paddingValue: Option[Array[T]] = None,
      paddingStrategy: Option[PaddingStrategy] = None)

abstract class PaddingStrategy {
  def padding(sizes: Seq[Array[Int]]): Seq[Array[Int]]
}

/**
 * Add an constant length to longest feature in the first dimension
 * @param paddingLength
 */
case class PaddingLongest(
      paddingLength: Array[Int]) extends PaddingStrategy {
    def padding(sizes: Seq[Array[Int]]): Seq[Array[Int]] = {
      var i = 0
      while (i < sizes.length) {
          // Add an constant length to the first dimension's length(besides mini batch size).
          val increment = paddingLength(i)
          sizes(i)(1) += increment
        i += 1
      }
      sizes
  }
}

/**
 * Set the first dimension's length to fixed length.
 * @param fixedLength fixed length
 */
case class FixedLength(fixedLength: Array[Int]) extends PaddingStrategy {
  def padding(sizes: Seq[Array[Int]]): Seq[Array[Int]] = {
    var i = 0
    while (i < sizes.length) {
        // Set the first dimension's length(besides mini batch size) to fixed length.
        val fixed = fixedLength(i)
        require(fixed >= sizes(i)(1) || fixed < 0,
          s"${i}th FixedLength=${fixed} is smaller than its FeatureLength=${sizes(i)(1)}")
        if (fixed >= sizes(i)(1)) {
          sizes(i)(1) = fixed
        }
      i += 1
    }
    sizes
  }
}
