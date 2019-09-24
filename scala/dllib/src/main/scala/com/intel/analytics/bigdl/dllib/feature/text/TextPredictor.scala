/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.text

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToMiniBatch, Transformer}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef

import scala.reflect.ClassTag

class TextPredictor[T: ClassTag](
    model: Module[T],
    batchPerThread: Int)(implicit ev: TensorNumeric[T]) extends Serializable {

  def predict(
    textSet: DistributedTextSet,
    shareBuffer: Boolean = false): DistributedTextSet = {
    TextPredictor.predict[T](textSet, model, batchPerThread, shareBuffer)
  }
}

object TextPredictor {
  def apply[T: ClassTag](
    model: Module[T],
    batchPerThread: Int)(implicit ev: TensorNumeric[T]): TextPredictor[T] = {
    new TextPredictor[T](model, batchPerThread)
  }

  def predict[T: ClassTag](
    textSet: DistributedTextSet,
    model: Module[T],
    batchPerThread: Int,
    shareBuffer: Boolean = false)(implicit ev: TensorNumeric[T]): DistributedTextSet = {
    val rdd = textSet.rdd
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model)
    val partitionNum = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch[T](
      batchSize = partitionNum * batchPerThread,
      partitionNum = Some(partitionNum)))
    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value()
      localModel.evaluate()
      val localToBatch = toBatchBroad.value.cloneTransformer()
      partition.grouped(batchPerThread).flatMap(textFeatures => {
        predictTextBatch[T](localModel, textFeatures, localToBatch, shareBuffer)
      })
    })
    TextSet.rdd(result).setWordIndex(textSet.getWordIndex)
  }

  def predictTextBatch[T: ClassTag](
      localModel: Module[T],
      textFeatures: Seq[TextFeature],
      localToBatch: Transformer[Sample[T], MiniBatch[T]],
      shareBuffer: Boolean = false)(implicit ev: TensorNumeric[T]): Seq[TextFeature] = {
    val samples = textFeatures.map(_.getSample.asInstanceOf[Sample[T]])
    val batchOutput = localToBatch(samples.toIterator).flatMap(batch => {
      val output = localModel.forward(batch.getInput()).toTensor[T]
      splitTensor[T](output, shareBuffer, batch.size())
    })
    textFeatures.toIterator.zip(batchOutput).foreach(zipped => {
      zipped._1(TextFeature.predict) = zipped._2
    })
    textFeatures
  }

  private def splitTensor[T: ClassTag](
     output: Tensor[T],
     shareBuffer: Boolean,
     batchSize: Int)(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    val result = if (shareBuffer) output else output.clone()
    val size = result.size(1)
    require(batchSize == size,
      s"batchSize is required to be $size, while the actual batchSize is $batchSize")
    result.split(1)
  }
}
