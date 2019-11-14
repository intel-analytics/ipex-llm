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

package com.intel.analytics.zoo.examples.localEstimator

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.ShortcutType
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim.{Adam, Loss, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.estimator.LocalEstimator
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class ResnetLocalEstimatorParams(imageDirPath: String = "./",
                                     batchSize: Int = 3 * 44,
                                     epoch: Int = 10,
                                     threadNum: Int = 44)

object ResnetLocalEstimator {

  val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]) : Unit = {
    val parser = new OptionParser[ResnetLocalEstimatorParams]("ResnetLocalEstimator Example") {
      opt[String]('d', "imageDirPath")
        .required()
        .text("The directory of mnist dataset")
        .action((x, c) => c.copy(imageDirPath = x))
      opt[Int]('b', "batchSize")
        .required()
        .text("The number of batchSize")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "epoch")
        .required()
        .text("The number of epoch")
        .action((x, c) => c.copy(epoch = x))
      opt[Int]('t', "threadNum")
        .required()
        .text("The number of threadNum")
        .action((x, c) => c.copy(threadNum = x))
    }

    parser.parse(args, ResnetLocalEstimatorParams()).map { params =>
      logger.info(s"params parsed as $params")

      val imageDirPath = params.imageDirPath
      val batchSize = params.batchSize
      val epoch = params.epoch
      val threadNum = params.threadNum

      val model: Module[Float] =
        ResNet.graph(10,
          T("shortcutType" -> ShortcutType.A, "depth" -> 50, "optnet" -> false))
      val criterion = new CrossEntropyCriterion[Float]()
      val adam = new Adam[Float]()
      val validations = Array(new Top1Accuracy[Float], new Top5Accuracy[Float], new Loss[Float])
      val localEstimator = LocalEstimator(model, criterion, adam, validations, threadNum)
      logger.info(s"LocalEstimator loaded as $localEstimator")

      val trainData = Cifar10DataLoader.loadTrainData(imageDirPath)
      val testData = Cifar10DataLoader.loadTestData(imageDirPath)
      localEstimator.fit(trainData,
        testData,
        ImageProcessing.labeledBGRImageToMiniBatchTransformer,
        batchSize,
        epoch)
    }
  }

}
