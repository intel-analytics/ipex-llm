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

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer, Utils}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

object BGRImgToBatch {
  def apply(batchSize: Int, toRGB: Boolean = true): BGRImgToBatch
    = new BGRImgToBatch(batchSize, toRGB)
}

/**
 * Convert a batch of labeled BGR images into a Mini-batch.
 *
 * Notice: The totalBatch means a total batch size. In distributed environment, the batch should be
 * divided by total core number
 * @param totalBatch
 * @param toRGB
 */
class BGRImgToBatch(totalBatch: Int, toRGB: Boolean = true)
  extends Transformer[LabeledBGRImage, MiniBatch[Float]] {

  private val batchPerCore = Utils.getBatchSize(totalBatch)

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[MiniBatch[Float]] = {
    val batchSizePerCore = batchPerCore

    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private var width = 0
      private var height = 0
      private val batchSize = batchSizePerCore

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val img = prev.next()
            if (featureData == null) {
              featureData = new Array[Float](batchSize * 3 * img.height() * img.width())
              labelData = new Array[Float](batchSize)
              height = img.height()
              width = img.width()
            }
            img.copyTo(featureData, i * img.width() * img.height() * 3, toRGB)
            labelData(i) = img.label()
            i += 1
          }

          if (labelTensor.nElement() != i) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, 3, height, width))
            labelTensor.set(Storage[Float](labelData),
              storageOffset = 1, sizes = Array(i))
          }

          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
