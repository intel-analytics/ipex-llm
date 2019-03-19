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

package com.intel.analytics.bigdl.example.mkldnnint8

import java.nio.file.Files

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.models.resnet.ImageNetDataSet
import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrameToSample, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
 * This example demonstrates how to evaluate pre-trained resnet-50 with ImageNet dataset using Int8
 */
object ImageNetInference {
  System.setProperty("bigdl.mkldnn.fusion", "true")

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  /**
    * if the model has not quantized and the backend is MklDnn, will generate the scales
    * and will save a model with a name including "quantized"
    *
    * @param sc spark context created in main
    * @param model a loaded BigDL model
    * @param batchSize batch size for scales generation
    * @param sampleFolder the samples folder
    * @param modelName model name you want
    */
  def generateScales(sc: SparkContext, model: Graph[Float], batchSize: Int,
    sampleFolder: String, modelName: String): Unit = {
    if (model.getOutputScales().isEmpty && Engine.getEngineType() == MklDnn) {
      logger.info(s"Generate the scales for $modelName ...")
      val partitionNum = Engine.nodeNumber()
      val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(sampleFolder, sc, 1000,
        partitionNum = Option(partitionNum))

      // the transformer is the same as as that in validation during training
      val transformer =
        PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, false, CropCenter) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float]() ->
          ImageFrameToSample[Float](inputKeys = Array("imageTensor"), targetKeys = Array("label"))

      imageFrame -> transformer

      val evaluateDataSet = imageFrame.toDistributed().rdd.map { x =>
        x[Sample[Float]](ImageFeature.sample)
      }

      val toMiniBatch = SampleToMiniBatch[Float](batchSize)
      val samples = evaluateDataSet
        .repartition(1)
        .mapPartitions(toMiniBatch(_))
        .take(1) // only split one or some batches to sample
        .map(_.getInput().toTensor[Float])

      // we current support the mask of input and output is 0,
      // and weight mask should is 1, which means the dimension of scales coming from.
      model.setInputDimMask(0)
      model.setOutputDimMask(0)
      model.setWeightDimMask(1)

      samples.foreach { sample =>
        model.calcScales(sample)
      }

      logger.info(s"Generate the scales for $modelName done.")

      val suffix = ".bigdl"
      val prefix = modelName.stripSuffix(suffix)
      val name = prefix.concat(".quantized").concat(suffix)
      logger.info(s"Save the quantized model $name ...")
      // it will force overWrite the existed model file
      model.saveModule(name, overWrite = true)
      logger.info(s"Save the quantized model $name done.")
    }
  }

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach(param => {
      val conf = Engine.createSparkConf().setAppName("Test model on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val model = Module.loadModule[Float](param.model).toGraph()
      generateScales(sc, model, param.batchSize, param.folder, param.model)

      logger.info(s"Quantize the model ...")
      val quantizedModel = model.quantize()
      logger.info(s"Quantize the model done.")

      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)

      val result = quantizedModel.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    })
  }
}
