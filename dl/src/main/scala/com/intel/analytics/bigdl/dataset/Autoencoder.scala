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
import com.intel.analytics.sparkdl.models.mnist.{LeNet5, MLP, SimpleCNN, AE}
import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Criterion, Module, TensorModule}
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{RandomGenerator, T}
import scopt.OptionParser

/**
  * This is an example program to demo how to use spark-dl to train nn model on MNIST dataset.
  * You can download the data from http://yann.lecun.com/exdb/mnist/
  */
object Autoencoder {
  case class AutoencoderParams(
                               folder: String = "./",
                               net: String = "ae"
                             )
  case class Config(
                     model : Module[Tensor[Float], Tensor[Float], Float],
                     criterion : Criterion[Tensor[Float], Float],
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
      new SGD[Float](), 10, 10, 0.05),
    "ae" -> Config(
      AE[Float](classNum = 32),
      new MSECriterion[Float](),
      new Adagrad[Float](), 150, 10, 0.001)
  )

  def main(args: Array[String]) {
    
      RandomGenerator.RNG.setSeed(1000)
      val trainData = "./train-images.idx3-ubyte"
      val trainDLabel = "./train-labels.idx1-ubyte"

      val trainDataSource = new MNISTDataSource(trainData, trainDLabel, looped = true)

      val arrayByteToImage = ArrayByteToGreyImage(28, 28)
      val normalizer = new GreyImageNormalizer(trainDataSource -> arrayByteToImage)
      val toAETensor = new GreyImageToAETensor(configs("ae").batchSize)
      val optimizer = new LocalOptimizer[Float](
        data = trainDataSource -> arrayByteToImage -> normalizer -> toAETensor,
        model = configs("ae").model,
        criterion = configs("ae").criterion,
        optimMethod = configs("ae").optimMethod,
        state = T("learningRate" -> configs("ae").learningRate),
        endWhen = Trigger.maxEpoch(configs("ae").maxEpoch)
      )
      optimizer.setValidationTrigger(Trigger.everyEpoch)
      optimizer.addValidation(new Top1Accuracy[Float])
      optimizer.optimize()
    
  }
}
