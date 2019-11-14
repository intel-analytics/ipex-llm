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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, Loss, Top1Accuracy}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.estimator.LocalEstimator
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class TransferLearningParams(imageDirPath: String = "./",
                                  modelPath: String = "./",
                                  inputs: String = "",
                                  outputs: String = "",
                                  batchSize: Int = 132,
                                  epoch: Int = 20,
                                  threadNum: Int = 10)

object TransferLearning {

  val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TransferLearningParams]("TransferLearning Example") {
      opt[String]('d', "imageDirPath")
        .required()
        .text("The directory of cifar10 dataset")
        .action((x, c) => c.copy(imageDirPath = x))
      opt[String]('m', "modelPath")
        .required()
        .text("The path of model")
        .action((x, c) => c.copy(modelPath = x))
      opt[String]('i', "inputs")
        .required()
        .text("The inputs of model")
        .action((x, c) => c.copy(inputs = x))
      opt[String]('o', "outputs")
        .required()
        .text("The outputs of model")
        .action((x, c) => c.copy(outputs = x))
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

    parser.parse(args, TransferLearningParams()).map { params =>
      logger.info(s"params parsed as $params")

      val modelPath = params.modelPath
      val inputs = params.inputs.split(",")
      val outputs = params.outputs.split(",")
      val imageDirPath = params.imageDirPath
      val batchSize = params.batchSize
      val epoch = params.epoch
      val threadNum = params.threadNum

      val originalModel = TFNet.fromSavedModel(modelPath, inputs, outputs)
      println(originalModel)

      val model = Sequential[Float]()
      model.add(Transpose[Float](Array((2, 4), (2, 3))))
      model.add(Contiguous[Float]())
      model.add(originalModel)
      model.add(Transpose[Float](Array((2, 4), (3, 4))))
      model.add(Contiguous[Float]())
      model.add(new SpatialAveragePooling[Float](2, 2, globalPooling = true))
      model.add(new View[Float](2048).setNumInputDims(3))
      model.add(new Linear[Float](2048, 2))

      println(model)

      val criterion = new CrossEntropyCriterion[Float]()
      val adam = new Adam[Float]()
      val validations = Array(new Top1Accuracy[Float], new Loss[Float])
      val localEstimator = LocalEstimator(model, criterion, adam, validations, threadNum)
      println(s"LocalEstimator loaded as $localEstimator")

      // use only a little data for transfer learning
      val trainData = Cifar10DataLoader.loadTrainData(imageDirPath)
        .filter(_.label() <= 2).slice(0, 10 * batchSize)
      val testData = Cifar10DataLoader.loadTestData(imageDirPath)
        .filter(_.label() <= 2).slice(0, 10 * batchSize)

      localEstimator.fit(trainData,
        testData,
        ImageProcessing.labeledBGRImageToMiniBatchTransformer,
        batchSize,
        epoch)
    }
  }

}
