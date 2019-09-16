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
package com.intel.analytics.bigdl.transform.vision.image

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer, Utils}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine

import scala.reflect.ClassTag

object MTImageFeatureToBatch {
  def apply(width: Int, height: Int, batchSize: Int,
            transformer: FeatureTransformer, toRGB: Boolean = true)
  : MTImageFeatureToBatch = {
    new MTImageFeatureToBatch (
      width, height, batchSize, transformer, toRGB)
  }
}

/**
 * An abstract class to convert ImageFeature iterator to MiniBatches. This transformer will run
 * on each image frame. "onEachImageFeature" will be called to buffer the image features. When there
 * are enough buffered image features to form a batch, "onEachBatch" will be called.
 * You should override onEachImageFeature to buffer each image feature, and onEachBatch
 * to convert the buffered data into a mini-batch
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 */
abstract class ImageFeatureToBatch private[bigdl](
  totalBatchSize: Int, transformer: FeatureTransformer)
  extends Transformer[ImageFeature, MiniBatch[Float]] {

  protected val batchSize: Int = Utils.getBatchSize(totalBatchSize)

  protected val parallelism: Int = Engine.coreNumber()

  private def getPosition(count: AtomicInteger): Int = {
    val position = count.getAndIncrement()
    if (position < batchSize) position else -1
  }

  private lazy val transformers = (1 to parallelism).map(
    _ => new PreFetch -> transformer.cloneTransformer()
  ).toArray

  protected def onEachImageFeature(img: ImageFeature, position: Int)

  protected def onEachBatch(batchSize: Int): MiniBatch[Float]

  override def apply(prev: Iterator[ImageFeature]): Iterator[MiniBatch[Float]] = {
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
            onEachImageFeature(img, position)
            record += 1
          }
          record
        })).sum
        onEachBatch(batch)
      }
    }
  }
}


private class PreFetch extends Transformer[ImageFeature, ImageFeature] {
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    new Iterator[ImageFeature] {
      private var buffer: ImageFeature = null.asInstanceOf[ImageFeature]

      override def hasNext: Boolean = {
        if (buffer != null) {
          true
        } else {
          buffer = prev.next()
          if (buffer == null) false else true
        }
      }

      override def next(): ImageFeature = {
        if (buffer == null) {
          prev.next()
        } else {
          val tmp = buffer
          buffer = null.asInstanceOf[ImageFeature]
          tmp
        }
      }
    }
  }
}

/**
 * A transformer pipeline wrapper to create Minibatch in multiple threads
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class MTImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = true)
  extends ImageFeatureToBatch(totalBatchSize, transformer) {

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  override protected def onEachImageFeature(img: ImageFeature, position: Int): Unit = {
    img.copyTo(featureData, position * frameLength * 3, toRGB = toRGB)
    labelData(position) = img.getLabel.asInstanceOf[Tensor[Float]].valueAt(1)
  }

  override protected def onEachBatch(batch: Int): MiniBatch[Float] = {
    if (labelTensor.nElement() != batch) {
      featureTensor.set(Storage[Float](featureData),
        storageOffset = 1, sizes = Array(batch, 3, height, width))
      labelTensor.set(Storage[Float](labelData),
        storageOffset = 1, sizes = Array(batch))
    }

    MiniBatch(featureTensor, labelTensor)
  }
}
