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
import com.intel.analytics.bigdl.dataset.{SampleToMiniBatch, _}
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.quantized.QuantizedModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.Util._
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn, Util}
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object LocalPredictor {

  val logger = Logger.getLogger(getClass)

  def apply[T: ClassTag](model: Module[T],
    featurePaddingParam: Option[PaddingParam[T]] = None,
    batchPerCore: Int = 4)
    (implicit ev: TensorNumeric[T]): LocalPredictor[T] = {
    new LocalPredictor[T](model, featurePaddingParam, batchPerCore)
  }
}

/**
 * Predictor for local data
 * @param model BigDL model
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 * @param batchPerCore batch size per core, default is 4
 */
class LocalPredictor[T: ClassTag] private[optim](model: Module[T],
  featurePaddingParam: Option[PaddingParam[T]] = None,
  batchPerCore: Int = 4)
  (implicit ev: TensorNumeric[T]) extends Serializable {

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case MklDnn => 1
  }

  private val batchPerModel = batchPerCore * coreNumber / subModelNumber

  // we should clone a new model which has no impact to origin model
  private val clonedModel = ConversionUtils.convert(model.cloneModule())

  private val workingModels = {

    val weightsBias = Util.getAndClearWeightBias(clonedModel.parameters())
    val models = (1 to subModelNumber).map(_ => {
      val submodel = clonedModel.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray
    Util.putWeightBias(weightsBias, clonedModel)
    Util.initGradWeightBias(weightsBias, clonedModel)
    models
  }

  val workingToBatch = {
    val toBatch = SampleToMiniBatch[T](
      batchSize = batchPerModel * subModelNumber,
      partitionNum = Some(subModelNumber),
      featurePaddingParam = featurePaddingParam)
    (1 to subModelNumber).map(_ => {
      toBatch.cloneTransformer()
    }).toArray
  }

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
    val dataIter = dataSet.grouped(batchPerModel * subModelNumber)

    dataIter.map(batch => {
      val groupedSamples = batch.grouped(batchPerModel).toArray
      Engine.default.invokeAndWait(
        groupedSamples.indices.map(b =>
          () => {
            val samples = groupedSamples(b)
            val model = workingModels(b)
            val toBatch = workingToBatch(b)
            Predictor.predictSamples(model, samples, toBatch, false)
          }
        )
      ).flatten
    }).flatten.toArray
  }

  /**
   * local model predict images, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   * outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: LocalImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): LocalImageFrame = {

    val dataIter = imageFrame.array.grouped(batchPerModel * subModelNumber)

    val result = dataIter.map(batch => {
      val groupedImages = batch.grouped(batchPerModel).toArray
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

  /**
   * `shutdown` will release all native resources.
   */
  def shutdown(): Unit = {
    workingModels.foreach(_.release())
    clonedModel.release()
  }
}


