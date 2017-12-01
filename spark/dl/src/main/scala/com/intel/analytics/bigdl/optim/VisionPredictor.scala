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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame}

import scala.reflect.ClassTag

class VisionPredictor[T: ClassTag](
  model: Module[T],
  postProcessor: PostProcessor = null)
  (implicit ev: TensorNumeric[T]) extends Serializable {

  def predict(imageFrame: ImageFrame,
    shareBuffer: Boolean = false,
    batchPerPartition: Int = 4): DistributedImageFrame = {
    require(imageFrame.isDistributed(), "please provide a distributed imageframe")
    val rdd = imageFrame.asInstanceOf[DistributedImageFrame].rdd
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model.evaluate())
    val partitionNum = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = partitionNum * batchPerPartition,
      partitionNum = Some(partitionNum)), shareBuffer)
    val postProcessorBroad = rdd.sparkContext.broadcast(postProcessor)
    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value()
      val localToBatch = toBatchBroad.value._1.cloneTransformer()
      val localPostProcessor = postProcessorBroad.value

      partition.grouped(batchPerPartition).flatMap(imageFeatures => {
        val validImageFeatures = imageFeatures.filter(_.isValid)
        val samples = validImageFeatures.map(x => x[Sample[T]](ImageFeature.sample))
        val batch = localToBatch(samples.toIterator).next()
        val result = localModel.forward(batch.getInput()).toTensor[T]
        val batchOut = if (result.dim() == 1) {
          Array(result)
        } else {
          result.split(1)
        }
        validImageFeatures.zip(batchOut).foreach(tuple => {
          tuple._1(ImageFeature.predict) = tuple._2
          if (localPostProcessor != null) {
            localPostProcessor.process(tuple._1)
          }
        })
        imageFeatures
      })
    })
    ImageFrame.rdd(result)
  }
}

trait PostProcessor extends Serializable {
  def process(imageFeature: ImageFeature): ImageFeature
}
