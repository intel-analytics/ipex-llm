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

package com.intel.analytics.bigdl.example.imageclassification.imageFrame

import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import com.intel.analytics.bigdl.example.loadmodel.ModelValidator.{TestLocalParams, logger, testLocalParser}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFrame, ImageFrameToSample, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import scopt.OptionParser

object InceptionValidation {
  case class ValidationParam(imageFolder: String = "",
                             modelPath: String = "",
                             batchSize: Int = 32)

  val parser = new OptionParser[ValidationParam]("Inception validation") {
    head("Inception validatio")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("modelPath")
      .text("BigDL model")
      .action((x, c) => c.copy(modelPath = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, ValidationParam()).foreach(param => {
      val conf = Engine.createSparkConf()
      conf.setAppName("BigDL ImageFrame API Example")
      val sc = new SparkContext(conf)
      Engine.init
      val imageFrame = SeqFileFolder.filesToImageFrame(param.imageFolder, sc, 1000)
      val model = Module.loadModule[Float](param.modelPath)
      val transformer = PixelBytesToMat() -> Resize(256, 256) ->
        CenterCrop(224, 224) -> ChannelNormalize(123, 117, 104) ->
        MatToTensor[Float]() ->
        ImageFrameToSample[Float](inputKeys = Array("imageTensor"),
          targetKeys = Array("label"))
      imageFrame -> transformer
      val result = model.evaluateImage(
        imageFrame,
        Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]()),
        Some(param.batchSize))
      result.foreach(r => {
        logger.info(s"${ r._2 } is ${ r._1 }")
      })
      sc.stop()
    })
  }
}
