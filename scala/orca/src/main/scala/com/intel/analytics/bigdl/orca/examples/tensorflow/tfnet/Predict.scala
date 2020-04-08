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

package com.intel.analytics.zoo.examples.tensorflow.tfnet

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(image: String = "/tmp/datasets/cat_dog/train_sampled",
                          model: String = "/tmp/models/ssd_mobilenet_v1_coco_2018_01_28" +
                            "/frozen_inference_graph.pb",
                          nPartition: Int = 4)

  val parser = new OptionParser[PredictParam]("TFNet Object Detection Example") {
    head("TFNet Object Detection Example")
    opt[String]('i', "image")
      .text("The path where the images are stored, can be either a folder or an image path")
      .action((x, c) => c.copy(image = x))
    opt[String]("model")
      .text("The path of the TensorFlow object detection model")
      .action((x, c) => c.copy(model = x))
    opt[Int]('p', "partition")
      .text("The number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>

      val sc = NNContext.initNNContext("TFNet Object Detection Example")

      val inputs = Array("image_tensor:0")
      val outputs = Array("num_detections:0", "detection_boxes:0",
        "detection_scores:0", "detection_classes:0")

      val model = TFNet(params.model, inputs, outputs)

      val data = ImageSet.read(params.image, sc, minPartitions = params.nPartition)
        .transform(
          ImageResize(256, 256) ->
            ImageMatToTensor(format = DataFormat.NHWC) ->
            ImageSetToSample())
      val output = model.predictImage(data.toImageFrame(), batchPerPartition = 1)

      // print the first result
      val result = output.toDistributed().rdd.first().predict()
      println(result)
      println("finished...")
      sc.stop()
    }
  }
}
