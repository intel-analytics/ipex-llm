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

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{T, Table}
import scala.collection.mutable.IndexedSeqView

object RoiImageFeatureToBatch {
  def apply(width: Int, height: Int, batchSize: Int,
    transformer: FeatureTransformer, toRGB: Boolean = true)
  : RoiImageFeatureToBatch = {
    new RoiImageFeatureToBatch (
      width, height, batchSize, transformer, toRGB)
  }
}

/**
 * A transformer pipeline wrapper to create RoiMiniBatch in multiple threads
 * The output "target" is a Table. The keys are from 1 to sizeof(batch). The values are
 * the fields of RoiLabel
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class RoiImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = true)
  extends ImageFeatureToBatch(totalBatchSize, transformer) {

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[RoiLabel] = new Array[RoiLabel](batchSize)
  private var featureTensor: Tensor[Float] = null


  override protected def onEachImageFeature(img: ImageFeature, position: Int): Unit = {
    img.copyTo(featureData, position * frameLength * 3, toRGB = toRGB)
    labelData(position) = img.getLabel.asInstanceOf[RoiLabel]
  }

  override protected def onEachBatch(batchSize: Int): MiniBatch[Float] = {
    if (featureTensor == null) {
      featureTensor = Tensor(Storage[Float](featureData),
        storageOffset = 1, size = Array(batchSize, 3, height, width))
    }
    RoiMiniBatch(featureTensor, labelData.view.view)
  }
}

class RoiMiniBatch(val input: Tensor[Float], val target: IndexedSeqView[RoiLabel,
  IndexedSeq[RoiLabel]])
  extends MiniBatch[Float] {

  override def size(): Int = {
    input.size(1)
  }

  override def getInput(): Tensor[Float] = input

  override def getTarget(): Table = T(target.map(_.toTable))

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.narrow(1, offset, length)
    val subTarget = target.view(offset - 1, length) // offset starts from 1
    RoiMiniBatch(subInput, subTarget)
  }

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float])
  : RoiMiniBatch.this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object RoiMiniBatch {
  def apply(data: Tensor[Float], target: IndexedSeqView[RoiLabel, IndexedSeq[RoiLabel]]):
  RoiMiniBatch = new RoiMiniBatch(data, target)
}
