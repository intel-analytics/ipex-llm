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

import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrameToSample, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object GenerateInt8Scales {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)


  import Utils._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Quantize the model")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      // We set partition number to be node*core, actually you can also assign other partitionNum
      val partitionNum = Engine.nodeNumber()
      val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(param.folder, sc, 1000,
        partitionNum = Option(partitionNum))
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

      val toMiniBatch = SampleToMiniBatch[Float](param.batchSize)
      val samples = evaluateDataSet
        .repartition(1)
        .mapPartitions(toMiniBatch(_))
        .take(1)
        .map(_.getInput().toTensor[Float])

      val model = Module.loadModule[Float](param.model).toGraph()

      model.evaluate()

      model.setInputDimMask(0)
      model.setOutputDimMask(0)
      model.setWeightDimMask(1)

      samples.foreach { sample =>
        model.forward(sample)
        model.calcScales(sample)
      }

      model.clearState()

      model.saveModule(path = param.model.concat(".quantized"), overWrite = true)

      sc.stop()
    }
  }
}
