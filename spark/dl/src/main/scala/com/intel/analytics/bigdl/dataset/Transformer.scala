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

      private val sampleData = new Array[Sample[T]](batchSize)
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
            if (null == sampleData(i)) {
              sampleData(i) = sample.clone()
            } else {
              sampleData(i).copy(sample)
            }
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
 */
private[bigdl] class SampleToMiniBatch[T: ClassTag](
    totalBatch: Int,
    miniBatch: Option[MiniBatch[T]] = None,
    paddingParam: Option[PaddingParam[T]] = None,
    partitionNum: Option[Int] = None)
    (implicit ev: TensorNumeric[T]) extends Transformer[Sample[T], MiniBatch[T]] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch)
  var miniBatchBuffer = miniBatch.orNull

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
          if (null == miniBatchBuffer) {
            miniBatchBuffer = MiniBatch(sampleData(0).numFeature(), sampleData(0).numLabel(),
              paddingParam)
          }

          if (i < batchSize) {
            miniBatchBuffer.setValue(sampleData.slice(0, i))
          } else {
            miniBatchBuffer.setValue(sampleData)
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
   * @param partitionNum partition number of dataset, default means partitionNum
   *                       equals Engine.nodeNumber()
   * @return
   */
  def apply[T: ClassTag](
        batchSize : Int,
        paddingParam: Option[PaddingParam[T]],
        partitionNum: Option[Int])(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, None, paddingParam, partitionNum)
  }

  def apply[T: ClassTag](
        batchSize : Int,
        miniBatch: MiniBatch[T],
        partitionNum: Option[Int])(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, Some(miniBatch), partitionNum = partitionNum)
  }

  def apply[T: ClassTag](
        batchSize : Int,
        featurePadding : Option[Tensor[T]] = None,
        labelPadding : Option[T] = None,
        fixedLength: Option[Int] = None,
        partitionNum: Option[Int] = None)(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    val fp = if (featurePadding.isDefined) Some(Array(featurePadding.get)) else None
    val fl = if (fixedLength.isDefined) Some(Array(fixedLength.get)) else None
    val lp = if (labelPadding.isDefined) Some(Array(labelPadding.get)) else None
    val ll = if (labelPadding.isDefined && fixedLength.isDefined) {
      Some(Array(fixedLength.get))
    } else {
      None
    }
    SampleToMiniBatch(batchSize, Some(PaddingParam[T](featurePadding = fp, featureFixedLength = fl,
      labelPadding = lp, labelFixedLength = ll)), partitionNum = partitionNum)
  }

//  /**
//   * Apply an SampleToMiniBatch transformer.
//   *
//   * @param batchSize total batch size
//   * @param toMiniBatch array[Sample] to MiniBatch function
//   * @return
//   */
//  def apply[T: ClassTag](
//        batchSize : Int,
//        miniBatch: MiniBatch[T]
//  )(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
//    new SampleToMiniBatch[T](batchSize, miniBatch)
//  }
//
//  /**
//   * Apply an SampleToMiniBatch transformer.
//   *
//   * @param batchSize total batch size
//   * @param toMiniBatch array[Sample] to MiniBatch function
//   * @param partitionNum partition number of dataset, default means partitionNum
//   *                     equals Engine.nodeNumber()
//   * @return
//   */
//  def apply[T: ClassTag](
//      batchSize : Int,
//      toMiniBatch : (Array[Sample[T]], Array[Tensor[T]], Array[Tensor[T]]) => MiniBatch[T],
//      partitionNum: Option[Int]
//  )(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
//    new SampleToMiniBatch[T](batchSize, partitionNum, toMiniBatch)
//  }
//
//  /**
//   * Apply an SampleToMiniBatch transformer.
//   *
//   * @param batchSize total batch size
//   * @param featurePadding feature padding value on the first feature tensor
//   *                       (by default None, meaning no feature padding)
//   * @param labelPadding label padding value (by default None, meaning no label padding)
//   * @param fixedLength if padding, it specifies the second dimension of feature/label
//   *                    after padding. If has multi feature tensor, only pad the first one.
//   *                    (by default None, meaning the length after padding is set to the max
//   *                    length of feature/label in a mini-batch)
//   * @param partitionNum partition number of dataset, default means partitionNum
//   *                     equals Engine.nodeNumber()
//   * @return
//   */
//  def apply[T: ClassTag](
//        batchSize : Int,
//        featurePadding : Option[Tensor[T]],
//        labelPadding : Option[T] = None,
//        fixedLength: Option[Int] = None,
//        partitionNum: Option[Int] = None)(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
//    val fp = if (featurePadding.isDefined) Some(Array(featurePadding.get)) else None
//    val lp = if (labelPadding.isDefined) Some(Array(labelPadding.get)) else None
//    val fl = if (fixedLength.isDefined) Some(Array(fixedLength.get)) else None
//    new SampleToMiniBatch(batchSize, featurePadding = fp, labelPadding = lp,
//      featureFixedLength = fl, partitionNum = partitionNum)
//  }

//  /**
//   * Convert a Array[Sample] to MiniBatch. This is the default toMiniBatch in SampleToMiniBatch.
//   *
//   * For example, we have 3 sample tensors, and convert them to a MiniBatch.
//   * Sample1's feature is a 2*3 tensor {1, 2, 3,
//   *                                    4, 5, 6}
//   *
//   * Sample2's feature is a 1*3 tensor {7, 8, 9}
//   *
//   * Sample3's feature is a 3*3 tensor {10, 11, 12,
//   *                                    13, 14, 15,
//   *                                    16, 17, 18}
//   *
//   * And the featurePadding is {0, 0, 0}, fixedLength is 4, the MiniBatch will be
//   * a tensor of 3*4*3:
//   * {1, 2, 3,
//   *  4, 5, 6,
//   *  0, 0, 0
//   *  0, 0, 0,
//   *
//   *  7, 8, 9,
//   *  0, 0, 0,
//   *  0, 0, 0,
//   *  0, 0, 0,
//   *
//   *  10, 11, 12,
//   *  13, 14, 15,
//   *  16, 17, 18
//   *  0, 0, 0}
//   *
//   *  Notice: If the sample has multi feature tensors, this function only pad the first one.
//   *
//   * @param samples inputs, a array of Sample
//   * @param buf1 input buffer, cache the data for input in MiniBatch.
//   * @param buf2 target buffer, cache the data for target in MiniBatch
//   * @param featurePadding feature padding value on the first feature tensor
//   *                       (by default None, meaning no feature padding)
//   * @param labelPadding label padding value (by default None, meaning no label padding)
//   * @param fixedLength if padding, it specifies the second dimension of feature/label
//   *                    after padding. If has multi feature tensor, only pad the first one.
//   *                    (by default None, meaning the length after padding is set to the max
//   *                    length of feature/label in a mini-batch)
//   * @param ev numeric operator
//   * @tparam T numeric type
//   * @return MiniBatch
//   */
}
