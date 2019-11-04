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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.models.resnet.Utils.{TestParams, _}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.utils.{Engine, T}
import scopt.OptionParser
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, segmentation}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.MeanAveragePrecision
import org.apache.spark.{SparkContext, rdd}
object Test {
  case class TestParams(
     folder: String = "./",
     model: String = "",
     batchSize: Int = 2,
     partitionNum: Int = -1
   )

  val testParser = new OptionParser[TestParams]("BigDL Mask-RCNN on COCO Test Example") {
    opt[String]('f', "folder")
      .text("the location of COCO dataset")
      .action((x, c) => c.copy(folder = x))

    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))

    opt[Int]('b', "batchSize")
      .text("total batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[Int]('p', "partitionNum")
      .text("partition number")
      .action((x, c) => c.copy(partitionNum = x))
  }

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param => {
      val conf = Engine.createSparkConf().setAppName("Test MaskRCNN on COCO")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val partitionNum = if (param.partitionNum > 0) param.partitionNum
      else Engine.nodeNumber() * Engine.coreNumber()

      val rddData = DataSet.SeqFileFolder.filesToRoiImageFeatures(param.folder,
        sc, Some(partitionNum))
        .toDistributed().data(train = false)

      val transformer = RoiImageFeatureToBatch.withResize(
        sizeDivisible = 32,
        batchSize = param.batchSize / Engine.nodeNumber(),
        transformer =
          PixelBytesToMat() ->
            ScaleResize(minSize = 800, maxSize = 1333) ->
            ChannelNormalize(122.7717f, 115.9465f, 102.9801f) ->
            MatToTensor[Float](),
            toRGB = false
        )
      val evaluationSet = transformer(rddData)

      val model = Module.loadModule[Float](param.model)

      val result = model.evaluate(evaluationSet,
        Array(MeanAveragePrecision.cocoBBox(81), MeanAveragePrecision.cocoSegmentation(81)))
      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    }}
  }
}
