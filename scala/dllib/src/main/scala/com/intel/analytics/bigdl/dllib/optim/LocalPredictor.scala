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
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}
import com.intel.analytics.bigdl.utils.Util._
import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, LocalImageFrame}

import scala.reflect.ClassTag

object LocalPredictor {

  def apply[T: ClassTag](model: Module[T], weightsBias: Array[Tensor[T]])
                        (implicit ev: TensorNumeric[T]): LocalPredictor[T] = {
    new LocalPredictor[T](model, weightsBias)
  }
}

class LocalPredictor[T: ClassTag] private[optim](model: Module[T], weightsBias: Array[Tensor[T]])
                                                (implicit ev: TensorNumeric[T])
  extends Serializable {

  val logger = LocalValidator.logger
  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  private val batchPerCore = 4

  def predictClass(dataSet: Array[Sample[T]]): Array[Int] = {
    val result = predict(dataSet)
    result.map(output => {
      val _output = output.toTensor[T]
      require(_output.dim() == 1, s"Predictor.predictClass:" +
        s"Only support one sample has one label, but got ${_output.dim()} label")
      ev.toType[Int](_output.max(1)._2.valueAt(1))
    })
  }

  def predictClass(dataSet: LocalDataSet[MiniBatch[T]]): Array[Int] = {
    val result = predict(dataSet)
    result.map(output => {
      val _output = output.toTensor[T]
      require(_output.dim() == 1, s"Predictor.predictClass:" +
        s"Only support one sample has one lable, but got ${_output.dim()} label")
      ev.toType[Int](_output.max(1)._2.valueAt(1))
    })
  }

  def predict(dataSet: LocalDataSet[MiniBatch[T]]): Array[Activity] = {
    val dataIter = dataSet.data(train = false)

    val workingModels = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray
    dataIter.map(batch => {
      println("Enter map")
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val output = workingModels(b).forward(input).toTensor[T]
            output
          }
        )
      )
      val batchResult = result.flatMap(_.split(1)).map(_.asInstanceOf[Activity])
      batchResult
    }).toArray.flatten

  }

  def predict(dataSet: Array[Sample[T]]): Array[Activity] = {
    val iter = dataSet.iterator
    val transformer = SampleToMiniBatch[T](
      batchSize = batchPerCore * subModelNumber, None, None,
      partitionNum = Some(1))
    val dataIter = transformer(iter)

    val workingModels = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray

    dataIter.map(batch => {
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val output = workingModels(b).forward(input).toTensor[T]
            output.clone()

          }
        )
      )
      val batchResult = result.flatMap(_.split(1)).map(_.asInstanceOf[Activity])
      batchResult
    }).toArray.flatten

  }

  /**
   * local model predict images, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   * outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param batchPerCore batch size per core, default is 4
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: LocalImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    batchPerCore: Int = 4,
    predictKey: String = ImageFeature.predict): LocalImageFrame = {

    val dataIter = imageFrame.array.grouped(batchPerCore * subModelNumber)

    val workingModels = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray

    // If batchPerCore == 1, will resize the feature every time in SampleToBatch
    def featurePaddingParam = if (batchPerCore == 1) Some(PaddingParam[T]()) else None

    val workingToBatch = (1 to subModelNumber).map(_ => {
      SampleToMiniBatch[T](
        batchSize = batchPerCore * subModelNumber,
        partitionNum = Some(subModelNumber),
        featurePaddingParam = featurePaddingParam)
    }).toArray

    val result = dataIter.map(batch => {
      val groupedImages = batch.grouped(batchPerCore).toArray
      Engine.default.invokeAndWait(
        groupedImages.indices.map(b =>
          () => {
            val imageFeatures = groupedImages(b)
            val model = workingModels(b)
            val toBatch = workingToBatch(b)
            Predictor.predictImageBatch[T](model, imageFeatures, outputLayer, predictKey,
              toBatch, shareBuffer)
          }
        )
      ).flatten
    }).flatten

    ImageFrame.array(result.toArray)
  }
}


