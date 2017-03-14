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

package com.intel.analytics.bigdl.dataset.image

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{Utils, MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import scala.reflect.ClassTag

object MTLabeledBGRImgToBatch {
  def apply[A: ClassTag](width: Int, height: Int, batchSize: Int,
    transformer: Transformer[A, LabeledBGRImage], toRGB: Boolean = true)
  : MTLabeledBGRImgToBatch[A] = {
    new MTLabeledBGRImgToBatch[A](
      width, height, batchSize, transformer, toRGB)
  }
}

/**
 * Multi-thread convert BGR images into Mini-Batch.
 *
 * Note: the source iterator must be thread safe.
 * @param width
 * @param height
 * @param totalBatchSize
 * @param transformer
 * @param toRGB
 * @tparam A
 */
class MTLabeledBGRImgToBatch[A: ClassTag] private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: Transformer[A, LabeledBGRImage], toRGB: Boolean = true)
  extends Transformer[A, MiniBatch[Float]] {

  private val batchSize = Utils.getBatchSize(totalBatchSize)

  private val parallelism = Engine.coreNumber()

  private def getPosition(count: AtomicInteger): Int = {
    val position = count.getAndIncrement()
    if (position < batchSize) position else -1
  }

  private lazy val transformers = (1 to parallelism).map(
    _ => new PreFetch[A] -> transformer.cloneTransformer()
  ).toArray

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  override def apply(prev: Iterator[A]): Iterator[MiniBatch[Float]] = {
    val iterators = transformers.map(_.apply(prev))

    new Iterator[MiniBatch[Float]] {
      override def hasNext: Boolean = {
        iterators.map(_.hasNext).reduce(_ || _)
      }

      override def next(): MiniBatch[Float] = {
        val count = new AtomicInteger(0)
        val batch = Engine.default.invokeAndWait((0 until parallelism).map(tid => () => {
          var position = 0
          var record = 0
          while (iterators(tid).hasNext && {
            position = getPosition(count)
            position != -1
          }) {
            val img = iterators(tid).next()
            img.copyTo(featureData, position * frameLength * 3, toRGB)
            labelData(position) = img.label()
            record += 1
          }
          record
        })).sum

        if (labelTensor.nElement() != batch) {
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(batch, 3, height, width))
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(batch))
        }

        MiniBatch(featureTensor, labelTensor)
      }
    }
  }
}

private class PreFetch[T] extends Transformer[T, T] {
  override def apply(prev: Iterator[T]): Iterator[T] = {
    new Iterator[T] {
      private var buffer: T = null.asInstanceOf[T]

      override def hasNext: Boolean = {
        if (buffer != null) {
          true
        } else {
          buffer = prev.next()
          if (buffer == null) false else true
        }
      }

      override def next(): T = {
        if (buffer == null) {
          prev.next()
        } else {
          val tmp = buffer
          buffer = null.asInstanceOf[T]
          tmp
        }
      }
    }
  }
}
