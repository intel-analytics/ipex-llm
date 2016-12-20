/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.image

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{Batch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import scala.reflect.ClassTag

object MTLabeledRGBImgToBatch {
  def apply[A: ClassTag](width: Int, height: Int, batchSize: Int,
    transformer: Transformer[A, LabeledRGBImage])
  : MTLabeledRGBImgToBatch[A] = {
    new MTLabeledRGBImgToBatch[A](
      width, height, batchSize, transformer)
  }
}

object MTLabeledRGBImgToBatchMultiNode {
  def apply[A: ClassTag](width: Int, height: Int, batchSize: Int,
    transformer: Transformer[A, LabeledRGBImage], nodeNumber : Int)
  : MTLabeledRGBImgToBatch[A] = {
    require(batchSize % nodeNumber == 0, "batch size can't be divided by node number")
    new MTLabeledRGBImgToBatch[A](
      width, height, batchSize / nodeNumber, transformer)
  }
}

class MTLabeledRGBImgToBatch[A: ClassTag] private[bigdl]
(width: Int, height: Int, batchSize: Int, transformer: Transformer[A, LabeledRGBImage])
  extends Transformer[A, Batch[Float]] {

  private def getPosition(count: AtomicInteger): Int = {
    val position = count.getAndIncrement()
    if (position < batchSize) position else -1
  }

  private lazy val transformers = (1 to Engine.coreNumber()).map(
    _ => new PreFetch[A] -> transformer.cloneTransformer()
  ).toArray

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  override def apply(prev: Iterator[A]): Iterator[Batch[Float]] = {
    val iterators = transformers.map(_.apply(prev))

    new Iterator[Batch[Float]] {
      override def hasNext: Boolean = {
        iterators.map(_.hasNext).reduce(_ || _)
      }

      override def next(): Batch[Float] = {
        val count = new AtomicInteger(0)
        val batch = Engine.default.invokeAndWait((0 until Engine.coreNumber()).map(tid => () => {
          var position = 0
          var record = 0
          while (iterators(tid).hasNext && {
            position = getPosition(count)
            position != -1
          }) {
            val img = iterators(tid).next()
            img.copyTo(featureData, position * frameLength * 3)
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

        Batch(featureTensor, labelTensor)
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
