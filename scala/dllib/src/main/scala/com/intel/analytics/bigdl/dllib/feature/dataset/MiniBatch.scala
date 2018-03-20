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
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
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
   *
   * @return size How many samples in this MiniBatch
   */
  def size(): Int

  /**
   * Slice this MiniBatch to a smaller MiniBatch with offset and length
   *
   * @param offset offset, counted from 1
   * @param length length
   * @return A smaller MiniBatch
   */
  def slice(offset: Int, length: Int): MiniBatch[T]

  /**
   * Get input in this MiniBatch.
   *
   * @return input Activity
   */
  def getInput(): Activity

  /**
   * Get target in this MiniBatch
   *
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
   *
   * @param samples a set of Sample
   * @return self
   */
  def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type
}

/**
 * Default type of MiniBatch in BigDL.
 * This MiniBatch support both single/multi inputs and single/multi targets.
 * `inputData` store the input tensors, if `inputData.length == 1`, `getInput()` will return
 * a tensor; If `inputData.length > 1`, `getInput()` will return a table.
 * `targetData` store the target tensors, if `targetData.length == 1`, `getTarget()` will return
 * a tensor; If `targetData.length > 1`, `getTarget()` will return a table.
 *
 * @param inputData           a set of input tensor
 * @param targetData          a set of target tensor
 * @param featurePaddingParam feature padding strategy, see
 *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
 * @param labelPaddingParam   label padding strategy, see
 *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
 * @tparam T Numeric type
 * @since 0.2.0
 */
private[bigdl] class ArrayTensorMiniBatch[T: ClassTag](
      val inputData: Array[Tensor[T]],
      val targetData: Array[Tensor[T]],
      featurePaddingParam: Option[PaddingParam[T]] = None,
      labelPaddingParam: Option[PaddingParam[T]] = None) extends MiniBatch[T]{
  require(inputData.length > 0, "Input data in MiniBatch is empty.")
  protected var batchSize = 0
  protected var unlabeled = false

  val (featurePadding, featurePaddingStrategy) = if (featurePaddingParam.isDefined) {
    (featurePaddingParam.get.paddingTensor, featurePaddingParam.get.paddingStrategy)
  } else {
    (None, new DefaultPadding)
  }

  val (labelPadding, labelPaddingStrategy) = if (labelPaddingParam.isDefined) {
    (labelPaddingParam.get.paddingTensor, labelPaddingParam.get.paddingStrategy)
  } else {
    (None, new DefaultPadding)
  }


  private val input: Activity = if (inputData.length == 1) {
    inputData.head
  } else {
    T.array(inputData.map(_.asInstanceOf[Any]))
  }

  private val target: Activity = if (targetData.length == 0) {
    null
  } else if (targetData.length == 1) {
    targetData.head
  } else {
    T.array(targetData.map(_.asInstanceOf[Any]))
  }

  override def size(): Int = {
    if (inputData.head.nElement() == 0) {
      0
    } else {
      inputData.head.size(1)
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

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0, "samples is empty")
    require(batchSize == 0 || samples.length <= batchSize, "setValue: samples's size doesn't " +
      s"match mini batch size, excepted ${size()} got ${samples.length}")
    val resize = batchSize != samples.length || featurePaddingParam.isDefined ||
      labelPaddingParam.isDefined || size() != samples.length
    if (batchSize == 0) {
      batchSize = samples.length // set a batchSize when set data.
      unlabeled = samples.head.numLabel() == 0
    }

    val longestFeature = if (featurePaddingParam.isDefined) {
      Some(MiniBatch.findLongestFeatures(samples))
    } else {
      None
    }

    val longestLabel = if (featurePaddingParam.isDefined) {
      Some(MiniBatch.findLongestLabels(samples))
    } else {
      None
    }

    if (resize) {
      MiniBatch.resize(samples, this, featurePaddingStrategy,
        labelPaddingStrategy, featurePadding, labelPadding,
        longestFeature, longestLabel)
    }

    MiniBatch.copyWithPadding[T](samples, this, unlabeled,
      featurePadding, labelPadding)
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
 *
   * @param nInputs number of inputs
   * @param nTargets number of targets
   * @return
   */
  def apply[T: ClassTag](
                          nInputs: Int,
                          nTargets: Int,
                          featurePaddingParam: Option[PaddingParam[T]] = None,
                          labelPaddingParam: Option[PaddingParam[T]] = None)(
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
        paddingStrategy: PaddingStrategy,
        paddingTensor: Option[Array[Tensor[T]]]): Unit = {
    // Size of input data. 1st Array is number of input, 2nd Array is input size.
    val sizes = new Array[Array[Int]](sampleSize.head.length)
    if (longestData.isDefined) {
      val longest = longestData.get

      var i = 0
      while (i < sizes.length) {
        // Set i-th input's size
        sizes(i) = Array(sampleSize.length) ++ sampleSize(longest(i))(i)
        i += 1
      }

      paddingStrategy.paddingSize(sizes)
    } else {
      var i = 0
      while (i < sizes.length) {
        // Set i-th input's size
        sizes(i) = Array(sampleSize.length) ++ sampleSize.head(i)
        i += 1
      }
    }

    // resize
    var i = 0
    while (i < sizes.length) {
      data(i).resize(sizes(i))
      if (paddingTensor.isEmpty) data(i).zero()
      i += 1
    }

  }

  // resize miniBatch, and zero miniBatch if paddingTensor is undefined.
  private[bigdl] def resize[T: ClassTag](
        samples: Seq[Sample[T]],
        miniBatch: ArrayTensorMiniBatch[T],
        featurePaddingStrategy: PaddingStrategy,
        labelPaddingStrategy: PaddingStrategy,
        featurePaddingTensor: Option[Array[Tensor[T]]] = None,
        labelPaddingTensor: Option[Array[Tensor[T]]] = None,
        longestFeature: Option[Array[Int]] = None,
        longestLabel: Option[Array[Int]] = None
      )(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = miniBatch.inputData
    val targets = miniBatch.targetData

    val featureSizes = samples.map(_.getFeatureSize())
    val unlabeled = samples.head.numLabel() == 0
    resizeData(inputs, featureSizes,
      longestFeature, featurePaddingStrategy, featurePaddingTensor)
    if (!unlabeled) {
      val labelSizes = samples.map(_.getLabelSize())
      resizeData(targets, labelSizes,
        longestLabel, labelPaddingStrategy, labelPaddingTensor)
    }

    miniBatch
  }

  private[bigdl] def copyWithPadding[T: ClassTag](
      samples: Seq[Sample[T]],
      miniBatch: ArrayTensorMiniBatch[T],
      unlabeled: Boolean,
      featurePadding: Option[Array[Tensor[T]]] = None,
      labelPadding: Option[Array[Tensor[T]]] = None
      )(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = miniBatch.inputData
    val targets = miniBatch.targetData

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

    // Copy sample data to miniBatch
    var s = 0
    while (s < samples.length) {
      var f = 0
      var offset = 0
      val sample = samples(s)
      val sampleData = sample.getData()
      while (f < inputs.length) {
        val length = sample.getFeatureSize()(f).product
        if (featurePadding.isDefined) {
          // copy data
          copy(sampleData, offset,
            length, inputs(f)(s + 1), featurePadding.get(f))
        } else {
          // copy data without padding.
          copy(sampleData, offset, length, inputs(f)(s + 1))
        }
        f += 1
        offset += length
      }

      if (!unlabeled) {
        var l = 0
        while (l < targets.length) {
          val length = sample.getLabelSize()(l).product
          if (labelPadding.isDefined) {
            // copy data
            copy(sampleData, offset,
              length, targets(l)(s + 1), labelPadding.get(l))
          } else {
            // copy data without padding.
            copy(sampleData, offset, length, targets(l)(s + 1))
          }
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
      new Array[Int](samples.head.numFeature())
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
      new Array[Int](samples.head.numLabel())
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
case class PaddingParam[T: ClassTag](
      paddingTensor: Option[Array[Tensor[T]]] = None,
      paddingStrategy: PaddingStrategy = new DefaultPadding) extends Serializable

abstract class PaddingStrategy extends Serializable {
  def paddingSize(sizes: Seq[Array[Int]]): Seq[Array[Int]]
}

class DefaultPadding extends PaddingStrategy {
  def paddingSize(sizes: Seq[Array[Int]]): Seq[Array[Int]] = {
    sizes
  }
}

/**
 * Add an constant length to longest feature in the first dimension
 *
 * @param paddingLength
 */
case class PaddingLongest(
      paddingLength: Array[Int]) extends PaddingStrategy {
    def paddingSize(sizes: Seq[Array[Int]]): Seq[Array[Int]] = {
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
 *
 * @param fixedLength fixed length
 */
case class FixedLength(fixedLength: Array[Int]) extends PaddingStrategy {
  def paddingSize(sizes: Seq[Array[Int]]): Seq[Array[Int]] = {
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

/**
 * SparseMiniBatch is a MiniBatch type for TensorSample. And SparseMiniBatch could
 * deal with SparseTensors in TensorSample.
 *
 * @param inputData           a set of input tensor
 * @param targetData          a set of target tensor
 * @param ev$1
 * @param ev
 * @tparam T Numeric type
 */
class SparseMiniBatch[T: ClassTag](
      inputData: Array[Tensor[T]],
      targetData: Array[Tensor[T]])(
      implicit ev: TensorNumeric[T]) extends ArrayTensorMiniBatch[T](inputData, targetData) {
  private var input: Activity = null
  private var target: Activity = null

  override def getInput(): Activity = {
    if (null == input) {
      require(!inputData.exists(_ == null), "SparseMiniBatch.getInput: " +
        "data didn't fill in this miniBatch")
      input = if (inputData.length == 1) {
        inputData.head
      } else {
        T.array(inputData.map(_.asInstanceOf[Any]))
      }
    }

    input
  }

  override def getTarget(): Activity = {
    if (null == target && targetData.length != 0) {
      require(!targetData.exists(_ == null), "SparseMiniBatch.getInput: " +
        "data didn't fill in this miniBatch")
      target = if (targetData.length == 1) {
        targetData.head
      } else {
        T.array(targetData.map(_.asInstanceOf[Any]))
      }
    }

    target
  }

  private def initTensor(sample: Tensor[_]): Tensor[_] = sample match {
    case s if s.getTensorType == SparseType =>
      s.getType() match {
        case tpe if tpe == BooleanType =>
          Tensor.sparse[Boolean](Array(batchSize) ++ s.size())
        case tpe if tpe == CharType =>
          Tensor.sparse[Char](Array(batchSize) ++ s.size())
        case tpe if tpe == StringType =>
          Tensor.sparse[String](Array(batchSize) ++ s.size())
        case tpe if tpe == IntType =>
          Tensor.sparse[Int](Array(batchSize) ++ s.size())
        case tpe if tpe == ShortType =>
          Tensor.sparse[Short](Array(batchSize) ++ s.size())
        case tpe if tpe == LongType =>
          Tensor.sparse[Long](Array(batchSize) ++ s.size())
        case tpe if tpe == FloatType =>
          Tensor.sparse[Float](Array(batchSize) ++ s.size())
        case tpe if tpe == DoubleType =>
          Tensor.sparse[Double](Array(batchSize) ++ s.size())
      }
    case s if s.getTensorType == DenseType =>
      s.getType() match {
        case tpe if tpe == BooleanType =>
          Tensor[Boolean](Array(batchSize) ++ s.size())
        case tpe if tpe == CharType =>
          Tensor[Char](Array(batchSize) ++ s.size())
        case tpe if tpe == StringType =>
          Tensor[String](Array(batchSize) ++ s.size())
        case tpe if tpe == IntType =>
          Tensor[Int](Array(batchSize) ++ s.size())
        case tpe if tpe == ShortType =>
          Tensor[Short](Array(batchSize) ++ s.size())
        case tpe if tpe == LongType =>
          Tensor[Long](Array(batchSize) ++ s.size())
        case tpe if tpe == FloatType =>
          Tensor[Float](Array(batchSize) ++ s.size())
        case tpe if tpe == DoubleType =>
          Tensor[Double](Array(batchSize) ++ s.size())
      }
    case s =>
      throw new IllegalArgumentException(s"MiniBatchWithSparse: unsupported feature type " +
        s"${s.getTensorType}")
  }

  def init(features: Array[Tensor[T]], labels: Array[Tensor[T]]): Unit = {
    features.zipWithIndex.foreach { case (feature, index) =>
      inputData(index) = initTensor(feature).asInstanceOf[Tensor[T]]
    }
    labels.zipWithIndex.foreach { case (label, index) =>
      targetData(index) = initTensor(label).asInstanceOf[Tensor[T]]
    }
  }

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0, "samples is empty")
    require(samples(0).isInstanceOf[TensorSample[T]])
    val _samples = samples.map(_.asInstanceOf[TensorSample[T]])
    require(batchSize == 0 || samples.length <= batchSize, "setValue: samples's size doesn't " +
      s"match mini batch size, excepted ${size()} got ${samples.length}")
    val features = _samples.map(_.features)
    val labels = _samples.map(_.labels)
    if (batchSize == 0) {
      batchSize = samples.length // set a batchSize when set data.
      unlabeled = samples.head.numLabel() == 0
      init(features.head, labels.head)
    }

    var i = 0
    while (i < inputData.length) {
      SparseMiniBatch.batch(1, features.map(_.apply(i)), inputData(i))
      i += 1
    }

    if (!unlabeled) {
      var j = 0
      while (j < targetData.length) {
        SparseMiniBatch.batch(1, labels.map(_.apply(j)), targetData(j))
        j += 1
      }
    }

    this
  }
}

object SparseMiniBatch{
  def apply[T: ClassTag](
      nInputs: Int,
      nTargets: Int)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    new SparseMiniBatch[T](new Array[Tensor[T]](nInputs), new Array[Tensor[T]](nTargets))
  }

  /**
   * Batch a seq of tensors to a big tensor.
   * @param dim apply batch on which dimension
   * @param tensors a seq of tensors
   * @param res result tensor
   * @param ev
   * @tparam T
   */
  private[bigdl] def batch[T: ClassTag](
      dim: Int,
      tensors: Seq[Tensor[T]],
      res: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    if (res.getTensorType == SparseType) {
      Tensor.sparseConcat(dim, tensors, res)
    } else if (res.getTensorType == DenseType) {
      denseBatch(dim, tensors, res)
    } else {
      throw new IllegalArgumentException(s"MiniBatchWithSparse: unsupported tensor type " +
        s"${res.getTensorType}")
    }
  }

  private def denseBatch[T: ClassTag](
        dim: Int,
        tensors: Seq[Tensor[T]],
        result: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val tensorSize = tensors.head.size()

    val (pre, next) = tensorSize.splitAt(dim - 1)
    val size = ArrayBuffer[Int]()
    size ++= pre
    size += tensors.length
    size ++= next

    result.resize(size.toArray)

    var i = 0
    while (i < tensors.length) {
      val current = tensors(i)
      val target = result.select(dim, i + 1)

      target.copy(current)

      i += 1
    }
    result

  }

}
