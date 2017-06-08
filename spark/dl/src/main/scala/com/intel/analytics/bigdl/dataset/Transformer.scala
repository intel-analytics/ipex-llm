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

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.apache.commons.lang3.SerializationUtils
import java.util

import com.intel.analytics.bigdl.utils.T
import org.apache.spark.rdd.RDD

import scala.collection.Iterator
import scala.reflect.ClassTag

/**
 * Transform a data stream of type A to type B. It is usually used in data pre-process stage.
 * Different transformers can compose a pipeline. For example, if there're transformer1 from A to
 * B, transformer2 from B to C, and transformer3 from C to D, you can compose them into a bigger
 * transformer from A to D by transformer1 -> transformer2 -> transformer 3.
 *
 * The purpose of transformer is for code reuse. Many deep learning share many common data
 * pre-process steps. User needn't write them every time, but can reuse others work.
 *
 * Transformer can be used with RDD(rdd.mapPartition), iterator and DataSet.
 * @tparam A
 * @tparam B
 */
trait Transformer[A, B] extends Serializable {
  def apply(prev: Iterator[A]): Iterator[B]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: Transformer[B, C]): Transformer[A, C] = {
    new ChainedTransformer(this, other)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  def cloneTransformer(): Transformer[A, B] = {
    SerializationUtils.clone(this)
  }

  /**
   * Apply this transformer to rdd
   * @param dataset
   */
  def apply(dataset: RDD[A])(implicit evidence: ClassTag[B]): RDD[B] = {
    val broadcast = dataset.sparkContext.broadcast(this)
    val cachedTransformer = dataset.mapPartitions(_ => Iterator
      .single(broadcast.value.cloneTransformer())
    ).setName("Transformer")

    dataset.zipPartitions(cachedTransformer)(
      (data, tran) => tran.next()(data))
  }
}

/**
 * A transformer chain two transformer together. The output type of the first transformer should be
 * same with the input type of the second transformer.
 * @param first first transformer
 * @param last last transformer
 * @tparam A input type of the first transformer
 * @tparam B output type of the first transformer, as well as the input type of the last transformer
 * @tparam C output of the last transformer
 */
class ChainedTransformer[A, B, C](first: Transformer[A, B], last: Transformer[B, C])
  extends Transformer[A, C] {
  override def apply(prev: Iterator[A]): Iterator[C] = {
    last(first(prev))
  }
}

object Identity {
  def apply[A](): Identity[A] = new Identity[A]()
}

/**
 * Just transform the input to output.
 */
class Identity[A] extends Transformer[A, A] {
  override def apply(prev: Iterator[A]): Iterator[A] = {
    prev
  }
}

/**
 * Convert a sequence of Sample to a sequence of MiniBatch,
 * optionally padding all the features (or labels) in the mini-batch to the same length
 */
object SampleToBatch {
  @deprecated("Use SampleToMiniBatch instead", "0.2.0")
  def apply[T: ClassTag]
  (batchSize : Int,
   featurePadding : Option[Tensor[T]] = None,
   labelPadding : Option[T] = None,
   fixedLength: Option[Int] = None,
   partitionNum: Option[Int] = None)
  (implicit ev: TensorNumeric[T]): SampleToBatch[T]
  = new SampleToBatch[T](batchSize, featurePadding, labelPadding, fixedLength, partitionNum)
}

/**
 * Convert a sequence of [[TensorSample]] to a sequence of [[TensorMiniBatch]],
 * optionally padding all the features (or labels) in the mini-batch to the same length
 *
 * @param totalBatch total batch size
 * @param featurePadding feature padding value (by default None, meaning no feature padding)
 * @param labelPadding label padding value (by default None, meaning no label padding)
 * @param fixedLength if padding, it specifies the length of feature/label after padding
 *                    (by default None, meaning the length after padding is set to the max
 *                    length of feature/label in a mini-batch)
 * @param partitionNum partition number of dataset, default means partitionNum
 *                     equals Engine.nodeNumber()
 */
@deprecated("Use SampleToMiniBatch instead", "0.2.0")
class SampleToBatch[T: ClassTag]
(totalBatch : Int,
 featurePadding : Option[Tensor[T]] = None,
 labelPadding : Option[T] = None,
 fixedLength: Option[Int] = None,
 partitionNum: Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  private def paddingTensor(data: Array[T], padValue: Tensor[T], start: Int, end: Int): Unit = {
    var offset = start
    val padArr = padValue.storage().array()
    val padOffset = padValue.storageOffset() - 1
    while (offset < end) {
      val length = math.min(end - offset, padArr.length)
      System.arraycopy(padArr, padOffset, data, offset, length)
      offset += length
    }
  }

  private def paddingValue(data: Array[T], padValue: T, start: Int, end: Int): Unit = {
    ev.getType() match {
      case DoubleType =>
        util.Arrays.fill(data.asInstanceOf[Array[Double]], start, end, ev.toType[Double](padValue))
      case FloatType =>
        util.Arrays.fill(data.asInstanceOf[Array[Float]], start, end, ev.toType[Float](padValue))
      case _ => throw new UnsupportedOperationException(
        "SampleToBatch: Only Float/Double supported")
    }
  }

  private def copyArray(
      src: Array[T],
      srcPos: Int,
      dest: Array[T],
      destPos: Int,
      length: Int): Unit = {
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

  /**
   * get data's product form start to end
   */
  private def getProduct(data: Array[Int], start: Int, end: Int): Int = {
    var i = start
    var res = 1
    while (i < end) {
      res *= data(i)
      i += 1
    }
    res
  }

  /**
   * compare a and b, then return the larger one's index
   * @param i the index of a
   * @param j the index of b
   */
  private def getLarger(a: Int, i : Int, b : Int, j : Int): Int = {
    if (a > b) i else j
  }

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    val batchSizePerPartition = batchPerPartition
    new Iterator[MiniBatch[T]] {
      private val featureTensor: Tensor[T] = Tensor[T]()
      private val labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private val batchSize = batchSizePerPartition

      private val sampleData = Array.tabulate(batchSize)(_ =>
        Sample(Tensor(), Tensor()))
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var oneFeatureElement: Int = 0
      private var oneLabelElement: Int = 0
      private val padFeature: Boolean = !featurePadding.isEmpty
      private val padLabel: Boolean = !labelPadding.isEmpty
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          var labelIndex = 0
          var featureIndex = 0
          var batchLength = 1
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            require(sample.feature().isContiguous() && sample.label().isContiguous(),
              "SampleToBatch: Only support contiguous tensor")
            sampleData(i).copy(sample)
            featureIndex = getLarger(sampleData(featureIndex).feature().nElement(),
              featureIndex, sample.feature().nElement(), i)
            labelIndex = getLarger(sampleData(labelIndex).label().nElement(),
              labelIndex, sample.label().nElement(), i)
            i += 1
          }
          batchLength = i
          if (featureSize == null) {
            featureSize = Array(1) ++ sampleData(featureIndex).feature().size()
            labelSize = Array(1) ++ sampleData(labelIndex).label().size()
          }

          featureSize(0) = batchLength
          val featureLength = sampleData(featureIndex).feature().size(1)
          featureSize(1) = if (padFeature) fixedLength.getOrElse(featureLength) else featureLength
          require(featureSize(1) >= featureLength,
            "SampleToBatch: fixedLength should not be less than first dimension of feature")
          oneFeatureElement = getProduct(featureSize, 1, featureSize.length)

          labelSize(0) = batchLength
          val labelLength = sampleData(labelIndex).label().size(1)
          labelSize(1) = if (padLabel) fixedLength.getOrElse(labelLength) else labelLength
          require(labelSize(1) >= labelLength,
            "SampleToBatch: fixedLength should not be less than first dimension of label")
          oneLabelElement = getProduct(labelSize, 1, labelSize.length)

          if (featureData == null || featureData.length < batchSize * oneFeatureElement) {
            featureData = new Array[T](batchSize * oneFeatureElement)
          }
          if (labelData == null || labelData.length < batchSize * oneLabelElement) {
            labelData = new Array[T](batchSize * oneLabelElement)
          }
          if (padFeature) {
            require(((featurePadding.get.dim() + 1) == sampleData(featureIndex).feature().dim())
              && featurePadding.get.isContiguous(), "SampleToBatch: featurePadding should be" +
              s"contiguous and dim should be ${sampleData(featureIndex).feature().dim() - 1}")
          }

          i = 0
          while (i < batchLength) {
            val sample = sampleData(i)
            copyArray(sample.feature().storage().array(), sample.feature().storageOffset() - 1,
              featureData, i * oneFeatureElement, sample.feature().nElement())
            if (padFeature) {
              paddingTensor(featureData, featurePadding.get,
                i * oneFeatureElement + sample.feature().nElement(), (i + 1) * oneFeatureElement)
            }
            copyArray(sample.label().storage().array(), sample.label().storageOffset() - 1,
              labelData, i * oneLabelElement, sample.label().nElement())
            if (padLabel) {
              paddingValue(labelData, labelPadding.get,
                i * oneLabelElement + sample.label().nElement(), (i + 1) * oneLabelElement)
            }
            i += 1
          }
          featureTensor.set(Storage[T](featureData), storageOffset = 1, sizes = featureSize)
          labelTensor.set(Storage[T](labelData), storageOffset = 1, sizes = labelSize)
          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}

/**
 * Convert a sequence of [[Sample]] to a sequence of [[MiniBatch]] through function toMiniBatch.
 *
 * @param totalBatch total batch size
 * @param partitionNum partition number of dataset, default means partitionNum
 *                     equals Engine.nodeNumber()
 * @param toMiniBatch toMiniBatch is an function convert an Array[Sample] to a MiniBatch[T], defined
 *                    as (Array[Sample[T]], Array[Tensor[T]], Array[Tensor[T]]) => MiniBatch[T]).
 *                    The two array[Tensor] are input buffers and target buffers, their lengths
 *                    equal to the Sample's numFeature and numLabel.
 */
class SampleToMiniBatch[T: ClassTag](
    totalBatch: Int,
    partitionNum: Option[Int] = None,
    toMiniBatch: (Array[Sample[T]], Array[Tensor[T]], Array[Tensor[T]]) => MiniBatch[T])
    (implicit ev: TensorNumeric[T]) extends Transformer[Sample[T], MiniBatch[T]] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch)
  private var inputBuffer: Array[Tensor[T]] = null
  private var targetBuffer: Array[Tensor[T]] = null

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    val batchSizePerPartition = batchPerPartition
    new Iterator[MiniBatch[T]] {
      private val batchSize = batchSizePerPartition

      private val sampleData = new Array[Sample[T]](batchSize)
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            if (null == sampleData(i)) {
              sampleData(i) = sample.clone()
            } else {
              sampleData(i).copy(sample)
            }
            i += 1
          }
          if (null == inputBuffer) {
            inputBuffer = Array.tabulate(sampleData(0).numFeature())(_ => Tensor[T]())
          }
          if (null == targetBuffer && sampleData(0).numLabel() > 0) {
            targetBuffer = Array.tabulate(sampleData(0).numLabel())(_ => Tensor[T]())
          }

          if (i == batchSize) {
            toMiniBatch(sampleData, inputBuffer, targetBuffer)
          } else {
            // Deal with number Sample is smaller than batchSize.
            toMiniBatch(sampleData.slice(0, i), inputBuffer, targetBuffer)
          }
        } else {
          null
        }
      }
    }
  }
}

object SampleToMiniBatch {
  /**
   * Apply an SampleToMiniBatch transformer.
   *
   * @param batchSize total batch size
   * @param toMiniBatch array[Sample] to MiniBatch function
   * @return
   */
  def apply[T: ClassTag](
        batchSize : Int,
        toMiniBatch : (Array[Sample[T]], Array[Tensor[T]], Array[Tensor[T]]) => MiniBatch[T]
  )(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, None, toMiniBatch)
  }

  /**
   * Apply an SampleToMiniBatch transformer.
   *
   * @param batchSize total batch size
   * @param toMiniBatch array[Sample] to MiniBatch function
   * @param partitionNum partition number of dataset, default means partitionNum
   *                     equals Engine.nodeNumber()
   * @return
   */
  def apply[T: ClassTag](
      batchSize : Int,
      toMiniBatch : (Array[Sample[T]], Array[Tensor[T]], Array[Tensor[T]]) => MiniBatch[T],
      partitionNum: Option[Int]
  )(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, partitionNum, toMiniBatch)
  }

  /**
   * Apply an SampleToMiniBatch transformer.
   *
   * @param batchSize total batch size
   * @param featurePadding feature padding value on the first feature tensor
   *                       (by default None, meaning no feature padding)
   * @param labelPadding label padding value (by default None, meaning no label padding)
   * @param fixedLength if padding, it specifies the second dimension of feature/label
   *                    after padding. If has multi feature tensor, only pad the first one.
   *                    (by default None, meaning the length after padding is set to the max
   *                    length of feature/label in a mini-batch)
   * @param partitionNum partition number of dataset, default means partitionNum
   *                     equals Engine.nodeNumber()
   * @return
   */
  def apply[T: ClassTag](
        batchSize : Int,
        featurePadding : Option[Tensor[T]] = None,
        labelPadding : Option[T] = None,
        fixedLength: Option[Int] = None,
        partitionNum: Option[Int] = None)(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    val toMiniBatch = (samples: Array[Sample[T]], b1: Array[Tensor[T]], b2: Array[Tensor[T]]) =>
      SampleToMiniBatch.samplesToMiniBatch[T](samples, b1, b2,
        featurePadding, labelPadding, fixedLength)
    SampleToMiniBatch(batchSize, toMiniBatch, partitionNum)
  }

  def apply[T: ClassTag](
    batchSize : Int,
    featurePaddings : Option[Array[Tensor[T]]],
    labelPadding : Option[T],
    partitionNum: Option[Int])(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    val toMiniBatch = (samples: Array[Sample[T]], b1: Array[Tensor[T]], b2: Array[Tensor[T]]) =>
      SampleToMiniBatch.arraySamplesToMiniBatch[T](samples, b1, b2,
        featurePaddings, labelPadding)
    SampleToMiniBatch(batchSize, toMiniBatch, partitionNum)
  }
  /**
   * Convert a Array[Sample] to MiniBatch. This is the default toMiniBatch in SampleToMiniBatch.
   *
   * For example, we have 3 sample tensors, and convert them to a MiniBatch.
   * Sample1's feature is a 2*3 tensor {1, 2, 3,
   *                                    4, 5, 6}
   *
   * Sample2's feature is a 1*3 tensor {7, 8, 9}
   *
   * Sample3's feature is a 3*3 tensor {10, 11, 12,
   *                                    13, 14, 15,
   *                                    16, 17, 18}
   *
   * And the featurePadding is {0, 0, 0}, fixedLength is 4, the MiniBatch will be
   * a tensor of 3*4*3:
   * {1, 2, 3,
   *  4, 5, 6,
   *  0, 0, 0
   *  0, 0, 0,
   *
   *  7, 8, 9,
   *  0, 0, 0,
   *  0, 0, 0,
   *  0, 0, 0,
   *
   *  10, 11, 12,
   *  13, 14, 15,
   *  16, 17, 18
   *  0, 0, 0}
   *
   *  Notice: If the sample has multi feature tensors, this function only pad the first one.
   *
   * @param samples inputs, a array of Sample
   * @param buf1 input buffer, cache the data for input in MiniBatch.
   * @param buf2 target buffer, cache the data for target in MiniBatch
   * @param featurePadding feature padding value on the first feature tensor
   *                       (by default None, meaning no feature padding)
   * @param labelPadding label padding value (by default None, meaning no label padding)
   * @param fixedLength if padding, it specifies the second dimension of feature/label
   *                    after padding. If has multi feature tensor, only pad the first one.
   *                    (by default None, meaning the length after padding is set to the max
   *                    length of feature/label in a mini-batch)
   * @param ev numeric operator
   * @tparam T numeric type
   * @return MiniBatch
   */
  def samplesToMiniBatch[T: ClassTag](
        samples: Array[Sample[T]],
        buf1: Array[Tensor[T]],
        buf2: Array[Tensor[T]] = null,
        featurePadding : Option[Tensor[T]] = None,
        labelPadding : Option[T] = None,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val featureIndex = if (featurePadding.isDefined) findLongestFeature(samples) else 0
    val labelIndex = if (labelPadding.isDefined) findLongestLabel(samples) else 0
    samples(0) match {
      case s: TensorSample[T] =>
        tensorSampleToMiniBatch(samples.map(_.asInstanceOf[TensorSample[T]]), buf1, buf2,
          featurePadding, featureIndex, labelPadding, labelIndex, fixedLength)
      case s: TensorTSample[T] =>
        tensorTSampleToMiniBatch(samples.map(_.asInstanceOf[TensorTSample[T]]), buf1, buf2,
          featurePadding, featureIndex, fixedLength)
      case s: ArrayTensorSample[T] =>
        arrayTensorSampleToMiniBatch(samples.map(_.asInstanceOf[ArrayTensorSample[T]]), buf1, buf2,
          featurePadding, featureIndex, labelPadding, labelIndex, fixedLength)
      case s: UnlabeledTensorSample[T] =>
        unlabeledTensorSampleToMiniBatch(samples.map(_.asInstanceOf[UnlabeledTensorSample[T]]),
          buf1, featurePadding, featureIndex, fixedLength)
      case s: UnlabeledArrayTensorSample[T] =>
        unlabeledArrayTensorSampleToMiniBatch(samples.map(
          _.asInstanceOf[UnlabeledArrayTensorSample[T]]), buf1, featurePadding,
          featureIndex, fixedLength)
      case _ => throw new IllegalArgumentException(s"toMiniBatch: " +
        s"Unsupported Sample type")
    }
  }

  /**
   * Find Sample in Array[Sample] who has the biggest featureLength
   */
  private def findLongestFeature[T: ClassTag](
        samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): Int = {
    var featureIndex = 0
    var i = 1
    while (i < samples.length) {
      if (samples(i).featureLength() > samples(featureIndex).featureLength()) {
        featureIndex = i
      }
      i += 1
    }
    featureIndex
  }

  /**
   * Find Sample in Array[Sample] who has the biggest labelLength
   */
  private def findLongestLabel[T: ClassTag](
        samples: Array[Sample[T]])(implicit ev: TensorNumeric[T]): Int = {
    var labelIndex = 0
    var i = 1
    while (i < samples.length) {
      if (samples(i).labelLength() > samples(labelIndex).labelLength()) {
        labelIndex = i
      }
      i += 1
    }
    labelIndex
  }

  def arraySamplesToMiniBatch[T: ClassTag](
    samples: Array[Sample[T]],
    buf1: Array[Tensor[T]],
    buf2: Array[Tensor[T]] = null,
    featurePaddings : Option[Array[Tensor[T]]] = None,
    labelPadding : Option[T] = None,
    fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val featureIndices =
      if (featurePaddings.isDefined) {
        findLongestFeatures(samples.map(_.asInstanceOf[ArrayTensorSample[T]]))
      } else {
        Array[Int]()
      }

    val labelIndex = if (labelPadding.isDefined) findLongestLabel(samples) else 0
    arrayTensorSampleToMiniBatch(samples.map(_.asInstanceOf[ArrayTensorSample[T]]), buf1, buf2,
      featurePaddings, featureIndices, labelPadding, labelIndex, fixedLength)
  }

  /**
   * Find Sample in Array[Sample] who has the biggest featureLength
   */
  private def findLongestFeatures[T: ClassTag](
    samples: Array[ArrayTensorSample[T]])(implicit ev: TensorNumeric[T]): Array[Int] = {
    val featureIndices =
      new Array[Int](samples(0).featuresLength().length).map(_ => 0)
    var i = 1
    while (i < samples.length) {
      var j = 0
      while (j < featureIndices.length) {
        if (samples(i).featuresLength()(j) > samples(featureIndices(j)).featuresLength()(j)) {
          featureIndices(j) = i
        }
        j += 1
      }
      i += 1
    }
    featureIndices
  }

  /**
   *  Convert an Array[TensorSample] to MiniBatch.
   */
  private def tensorSampleToMiniBatch[T: ClassTag](
        samples: Array[TensorSample[T]],
        buf1: Array[Tensor[T]],
        buf2: Array[Tensor[T]],
        featurePadding : Option[Tensor[T]] = None,
        featureIndex: Int = 0,
        labelPadding : Option[T] = None,
        labelIndex: Int = 0,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val input = buf1(0)
    val target = buf2(0)

    // Compute the input's Size and target's Size
    val inputSize = Array(samples.length) ++ samples(featureIndex).featureTensor.size()
    val targetSize = Array(samples.length) ++ samples(labelIndex).labelTensor.size()
    if (fixedLength.isDefined) {
      require(fixedLength.get >= inputSize(1), "FixedLength is smaller than feature length")
      if (featurePadding.isDefined) inputSize(1) = fixedLength.get
      if (labelPadding.isDefined) targetSize(1) = fixedLength.get
    }

    // Resize the input and target to right size.
    input.resize(inputSize)
    target.resize(targetSize)

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      require(featurePadding.get.dim() == input.dim() - 2, "featurePadding should have the " +
        s"same dimension with the feature in sample. Excepted: ${input.dim() - 2}, " +
        s"but got ${featurePadding.get.dim()}")
      require(featurePadding.get.isContiguous(), "featurePadding should be contiguous")
    }

    if (labelPadding.isDefined) {
      // fill target with labelPadding first.
      target.fill(labelPadding.get)
    }

    // Copy sample data to miniBatch
    var i = 1
    while (i <= samples.length) {
      if (featurePadding.isDefined) {
        // copy data with padding
        copyWithPadding(samples(i - 1).featureTensor, input(i), featurePadding.get)
      } else {
        // copy data without padding.
        input(i).copy(samples(i - 1).featureTensor)
      }

      if (labelPadding.isDefined) {
        // copy data only, as target has been filled by labelPadding.
        copy(samples(i - 1).labelTensor, target(i))
      } else {
        // copy data without padding.
        target(i).copy(samples(i - 1).labelTensor)
      }

      i += 1
    }

    MiniBatch(input, target)
  }

  /**
   *  Convert an Array[TensorTSample] to MiniBatch.
   */
  private def tensorTSampleToMiniBatch[T: ClassTag](
        samples: Array[TensorTSample[T]],
        buf1: Array[Tensor[T]],
        buf2: Array[Tensor[T]],
        featurePadding : Option[Tensor[T]] = None,
        featureIndex: Int = 0,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val input = buf1(0)
    val target = buf2(0)

    // Compute the input's Size and target's Size
    val inputSize = Array(samples.length) ++ samples(featureIndex).featureTensor.size()
    val targetSize = Array(samples.length)
    if (fixedLength.isDefined) {
      require(fixedLength.get >= inputSize(1), "FixedLength is smaller than feature length")
      inputSize(1) = fixedLength.get
    }

    // Resize the input and target to right size.
    input.resize(inputSize)
    target.resize(targetSize)

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      require(featurePadding.get.dim() == input.dim() - 2, "featurePadding should have the " +
        s"same dimension with the feature in sample. Excepted: ${input.dim() - 2}, " +
        s"but got ${featurePadding.get.dim()}")
      require(featurePadding.get.isContiguous(), "featurePadding should be contiguous")
    }

    // Copy sample data to miniBatch
    var i = 1
    while (i <= samples.length) {
      if (featurePadding.isDefined) {
        copyWithPadding(samples(i - 1).featureTensor, input(i), featurePadding.get)
      } else {
        // copy data without padding.
        input(i).copy(samples(i - 1).featureTensor)
      }

      target.setValue(i, samples(i - 1).labelValue)

      i += 1
    }

    MiniBatch(input, target)
  }

  /**
   *  Convert an Array[ArrayTensorSample] to MiniBatch.
   */
  def arrayTensorSampleToMiniBatch[T: ClassTag](
        samples: Array[ArrayTensorSample[T]],
        buf1: Array[Tensor[T]],
        buf2: Array[Tensor[T]],
        featurePadding : Option[Tensor[T]] = None,
        featureIndex: Int = 0,
        labelPadding : Option[T] = None,
        labelIndex: Int = 0,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = buf1
    val target = buf2(0)

    // Compute the input's Size and target's Size
    val input1Size = Array(samples.length) ++ samples(featureIndex).features(0).size()
    val targetSize = Array(samples.length) ++ samples(labelIndex).labels.size()
    if (fixedLength.isDefined) {
      require(fixedLength.get >= input1Size(2), "FixedLength is smaller than feature length")
      if (featurePadding.isDefined) input1Size(1) = fixedLength.get
      if (labelPadding.isDefined) targetSize(1) = fixedLength.get
    }

    // Resize the input and target to right size.
    inputs(0).resize(input1Size)
    var i = 1
    while (i < samples(featureIndex).features.length) {
      inputs(i).resize(Array(samples.length) ++ samples(featureIndex).features(i).size())
      i += 1
    }
    target.resize(targetSize)

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      require(featurePadding.get.dim() == inputs(0).dim() - 2, "featurePadding should have the " +
        s"same dimension with the feature in sample. Excepted: ${inputs(0).dim() - 2}, " +
        s"but got ${featurePadding.get.dim()}")
      require(featurePadding.get.isContiguous(), "featurePadding should be contiguous")
    }

    if (labelPadding.isDefined) {
      // fill target with labelPadding first.
      target.fill(labelPadding.get)
    }

    // Copy sample data to miniBatch
    var s = 1
    while (s <= samples.length) {
      if (featurePadding.isDefined) {
        // copy data
        copyWithPadding(samples(s - 1).features(0), inputs(0)(s), featurePadding.get)
      } else {
        // copy data without padding.
        inputs(0)(s).copy(samples(s - 1).features(0))
      }

      i = 1
      while (i < samples(s - 1).features.length) {
        inputs(i)(s).copy(samples(s - 1).features(i))
        i += 1
      }

      if (labelPadding.isDefined) {
        // copy data only, as target has been filled by labelPadding.
        copy(samples(s - 1).labels, target(s))
      } else {
        // copy data without padding.
        target(s).copy(samples(s - 1).labels)
      }

      s += 1
    }

    // inputs Array[Tensor] to table
    val input = T()
    i = 1
    while(i <= inputs.length) {
      input(i) = inputs(i - 1)
      i += 1
    }

    MiniBatch(input, target)
  }


  /**
   *  Convert an Array[ArrayTensorSample] to MiniBatch.
   */
  def arrayTensorSampleToMiniBatch[T: ClassTag](
    samples: Array[ArrayTensorSample[T]],
    buf1: Array[Tensor[T]],
    buf2: Array[Tensor[T]],
    featurePaddings : Option[Array[Tensor[T]]],
    featureIndices: Array[Int],
    labelPadding : Option[T],
    labelIndex: Int,
    fixedLength: Option[Int])(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = buf1
    val target = buf2(0)

    // Compute the input's Size and target's Size
    val input1Size = Array(samples.length) ++ samples(featureIndices(0)).features(0).size()
    val targetSize = Array(samples.length) ++ samples(labelIndex).labels.size()
    if (fixedLength.isDefined) {
      require(fixedLength.get >= input1Size(2), "FixedLength is smaller than feature length")
      if (featurePaddings.isDefined) input1Size(1) = fixedLength.get
      if (labelPadding.isDefined) targetSize(1) = fixedLength.get
    }

    // Resize the input and target to right size.
    inputs(0).resize(input1Size)
    var i = 1
    while (i < samples(0).features.length) {
      inputs(i).resize(Array(samples.length) ++ samples(0).features(i).size())
      i += 1
    }
    target.resize(targetSize)

    if (labelPadding.isDefined) {
      // fill target with labelPadding first.
      target.fill(labelPadding.get)
    }

    // Copy sample data to miniBatch
    var s = 1
    while (s <= samples.length) {
      i = 0
      while (i < samples(s - 1).features.length) {
        if (featurePaddings.isDefined) {
          // check if featurePadding is right.
          require(featurePaddings.get(i).dim() == inputs(0).dim() - 2,
            "featurePadding should have the " +
            s"same dimension with the feature in sample. Excepted: ${inputs(0).dim() - 2}, " +
            s"but got ${featurePaddings.get(i).dim()}")
          require(featurePaddings.get(i).isContiguous(), "featurePadding should be contiguous")
          // copy data
          copyWithPadding(samples(s - 1).features(i), inputs(i)(s), featurePaddings.get(i))
        } else {
          // copy data without padding.
          inputs(i)(s).copy(samples(s - 1).features(i))
        }
        i += 1
      }

      if (labelPadding.isDefined) {
        // copy data only, as target has been filled by labelPadding.
        copy(samples(s - 1).labels, target(s))
      } else {
        // copy data without padding.
        target(s).copy(samples(s - 1).labels)
      }

      s += 1
    }

    // inputs Array[Tensor] to table
    val input = T()
    i = 1
    while(i <= inputs.length) {
      input(i) = inputs(i - 1)
      i += 1
    }

    MiniBatch(input, target)
  }

  /**
   *  Convert an Array[UnlabeledArrayTensorSample] to MiniBatch.
   */
  def unlabeledArrayTensorSampleToMiniBatch[T: ClassTag](
        samples: Array[UnlabeledArrayTensorSample[T]],
        buf1: Array[Tensor[T]],
        featurePadding : Option[Tensor[T]] = None,
        featureIndex: Int = 0,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {

    val inputs = buf1

    // Compute the input's Size
    val input1Size = Array(samples.length) ++ samples(featureIndex).features(0).size()
    if (fixedLength.isDefined) {
      require(fixedLength.get >= input1Size(2), "FixedLength is smaller than feature length")
      input1Size(1) = fixedLength.get
    }

    // Resize the input to right size.
    inputs(0).resize(input1Size)
    var i = 1
    while (i < samples(featureIndex).features.length) {
      inputs(i).resize(Array(samples.length) ++ samples(featureIndex).features(i).size())
      i += 1
    }

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      require(featurePadding.get.dim() == inputs(0).dim() - 2, "featurePadding should have the " +
        s"same dimension with the feature in sample. Excepted: ${inputs(0).dim() - 2}, " +
        s"but got ${featurePadding.get.dim()}")
      require(featurePadding.get.isContiguous(), "featurePadding should be contiguous")
    }

    // Copy sample data to miniBatch
    var s = 1
    while (s <= samples.length) {
      if (featurePadding.isDefined) {
        copyWithPadding(samples(s - 1).features(0), inputs(0)(s), featurePadding.get)
      } else {
        // copy data without padding.
        inputs(0)(s).copy(samples(s - 1).features(0))
      }

      i = 1
      while (i < samples(s - 1).features.length) {
        inputs(i)(s).copy(samples(s - 1).features(i))
        i += 1
      }

      s += 1
    }

    // inputs Array[Tensor] to table
    val input = T()
    i = 1
    while(i <= inputs.length) {
      input(i) = inputs(i - 1)
      i += 1
    }

    MiniBatch(input)
  }

  /**
   *  Convert an Array[UnlabeledTensorSample] to MiniBatch.
   */
  def unlabeledTensorSampleToMiniBatch[T: ClassTag](
        samples: Array[UnlabeledTensorSample[T]],
        buf1: Array[Tensor[T]],
        featurePadding : Option[Tensor[T]] = None,
        featureIndex: Int = 0,
        fixedLength: Option[Int] = None)(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val input = buf1(0)

    // Compute the input's Size
    val inputSize = Array(samples.length) ++ samples(featureIndex).featureTensor.size()
    if (fixedLength.isDefined) {
      require(fixedLength.get >= inputSize(1), "FixedLength is smaller than feature length")
      inputSize(1) = fixedLength.get
    }

    // Resize the input to right size.
    input.resize(inputSize)

    if (featurePadding.isDefined) {
      // check if featurePadding is right.
      require(featurePadding.get.dim() == input.dim() - 2, "featurePadding should have the " +
        s"same dimension with the feature in sample. Excepted: ${input.dim() - 2}, " +
        s"but got ${featurePadding.get.dim()}")
      require(featurePadding.get.isContiguous(), "featurePadding should be contiguous")
    }

    // Copy sample data to miniBatch
    var i = 1
    while (i <= samples.length) {
      if (featurePadding.isDefined) {
        copyWithPadding(samples(i - 1).featureTensor, input(i), featurePadding.get)
      } else {
        // copy data without padding.
        input(i).copy(samples(i - 1).featureTensor)
      }

      i += 1
    }

    MiniBatch(input)
  }

  /**
   * Copy tensor src to tensor dest.
   */
  private def copy[T: ClassTag](
        src: Tensor[T],
        dest: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    require(src.isContiguous() && dest.isContiguous(), "src and dest should be contiguous")
    arrayCopy(src.storage.array(),
      src.storageOffset() - 1,
      dest.storage().array(),
      dest.storageOffset() - 1,
      src.nElement())
  }

  /**
   * Copy tensor src to tensor dest with a padding tensor.
   */
  private def copyWithPadding[T: ClassTag](
        src: Tensor[T],
        dest: Tensor[T],
        paddingTensor: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    arrayCopy(src.storage.array(),
      src.storageOffset() - 1,
      dest.storage().array(),
      dest.storageOffset() - 1,
      src.nElement())
    var j = src.size(1) + 1
    while (j <= dest.size(1)) {
      dest(j).copy(paddingTensor)
      j += 1
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
