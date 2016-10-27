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

import com.intel.analytics.sparkdl.example.MNIST
import com.intel.analytics.sparkdl.models.mnist.{LeNet5, MLP, SimpleCNN}
import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Criterion, Module, TensorModule}
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{RandomGenerator, T}
import scopt.OptionParser

/**
 * This is an example program to demo how to use spark-dl to train nn model on MNIST dataset.
 * You can download the data from http://yann.lecun.com/exdb/mnist/
 */
object MNISTLocal {
  case class MNISTLocalParams(
    folder: String = "./",
    net: String = "cnn"
  )
  case class Config(
    model : Module[Tensor[Float], Tensor[Float], Float],
    criterion : Criterion[Tensor[Float], Tensor[Float], Float],
    optimMethod : OptimMethod[Float],
    batchSize : Int,
    maxEpoch : Int,
    learningRate : Double
  )

  private val configs = Map(
    "mlp" -> Config(
      MLP[Float](classNum = 10),
      new ClassNLLCriterion[Float](),
      new SGD[Float](), 10, 10, 0.05),
    "cnn" -> Config(
      SimpleCNN[Float](classNum = 10),
      new ClassNLLCriterion[Float](),
      new SGD[Float](), 10, 10, 0.05),
    "lenet" -> Config(
      LeNet5[Float](classNum = 10),
      new ClassNLLCriterion[Float](),
      new SGD[Float](), 10, 10, 0.05)
  )

  private val parser = new OptionParser[MNISTLocalParams]("Spark-DL MNIST Local Example") {
    head("Spark-DL MNIST Local Example")
    opt[String]('f', "folder")
      .text("where you put the MNIST data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : mlp | cnn | lenet")
      .action((x, c) => c.copy(net = x.toLowerCase))
      .validate(v =>
        if (Set("mlp", "cnn", "lenet").contains(v.toLowerCase())) {
          success
        } else {
          failure("Net type can only be mlp | cnn | lenet in this example")
        }
      )
  }

  def main(args: Array[String]) {
    parser.parse(args, new MNISTLocalParams()).map(param => {
      RandomGenerator.RNG.setSeed(1000)
      val trainData = param.folder + "/train-images.idx3-ubyte"
      val trainDLabel = param.folder + "/train-labels.idx1-ubyte"
      val validationData = param.folder + "/t10k-images.idx3-ubyte"
      val validationLabel = param.folder + "/t10k-labels.idx1-ubyte"

      val trainDataSource = new MNISTDataSource(trainData, trainDLabel, looped = true)
      val validationDataSource = new MNISTDataSource(validationData, validationLabel, looped =
        false)
      val normalizer = new GreyImageNormalizer(trainDataSource)
      val toTensor = new GreyImageToTensor(configs(param.net).batchSize)
      val optimizer = new LocalOptimizer[Float](
        data = trainDataSource ++ normalizer ++ toTensor,
        validationData = validationDataSource ++ normalizer ++ toTensor,
        model = configs(param.net).model,
        criterion = configs(param.net).criterion,
        optimMethod = configs(param.net).optimMethod,
        state = T("learningRate" -> configs(param.net).learningRate),
        endWhen = Trigger.maxEpoch(configs(param.net).maxEpoch)
      )
      optimizer.setValidationTrigger(Trigger.everyEpoch)
      optimizer.addValidation(new Top1Accuracy[Float])
      optimizer.optimize()
    })
  }
}
