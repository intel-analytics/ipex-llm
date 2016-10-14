/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.dataset

import java.nio.file.Paths

import com.intel.analytics.sparkdl.models.imagenet.{AlexNet, GoogleNet_v1}
import com.intel.analytics.sparkdl.nn.ClassNLLCriterion
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.utils.T
import scopt.OptionParser

object ImageNetLocal {
  case class ImageNetLocalParam(
    folder: String = "./",
    net: String = "alexnet",
    cache: String = "./"
  )

  private val parser = new OptionParser[ImageNetLocalParam]("Spark-DL ImageNet Local Example") {
    head("Spark-DL ImageNet Local Example")
    opt[String]('f', "folder")
      .text("where you put the ImageNet data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('c', "cache")
      .text("where you put the model and state snapshot")
      .action((x, c) => c.copy(cache = x))
    opt[String]('n', "net")
      .text("net type : alexnet | googlenetv1")
      .action((x, c) => c.copy(net = x.toLowerCase))
      .validate(v =>
        if (Set("alexnet", "googlenetv1").contains(v.toLowerCase())) {
          success
        } else {
          failure("Net type can only be alexnet | googlenetv1 in this example")
        }
      )
  }

  def main(args: Array[String]) {
    parser.parse(args, new ImageNetLocalParam()).map(param => {
      val trainDataSource = new ImageNetDataSource(Paths.get(param.folder + "/train"),
        looped = true)
      val validationDatasource = new ImageNetDataSource(Paths.get(param.folder + "/val"),
        looped = false)
      val cropper = new RGBImageCropper(cropWidth = 224, cropHeight = 224)
      val normalizer = new RGBImageNormalizer(trainDataSource)
      val toTensor = new RGBImageToTensor(batchSize = 10)
      val model = param.net match {
        case "alexnet" => AlexNet[Float](classNum = 1000)
        case "googlenetv1" => GoogleNet_v1[Float](classNum = 1000)
        case _ => throw new IllegalArgumentException
      }

      val optimizer = new LocalOptimizer[Float](
        data = trainDataSource ++ cropper ++ normalizer ++ toTensor,
        validationData = validationDatasource ++ cropper ++ normalizer ++ toTensor,
        model = model,
        criterion = new ClassNLLCriterion[Float](),
        optimMethod = new SGD[Float](),
        state = T("learningRate" -> 0.05),
        endWhen = Trigger.maxEpoch(2)
      )
      optimizer.setCache(param.cache, Trigger.everyEpoch)
      optimizer.setValidationTrigger(Trigger.everyEpoch)
      optimizer.addValidation(new Top1Accuracy[Float])
      optimizer.addValidation(new Top5Accuracy[Float])
      optimizer.optimize()
    })
  }

}
