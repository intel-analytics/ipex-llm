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

package com.intel.analytics.bigdl.example.tensorflow.transferlearning

import java.nio.ByteOrder

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.utils.tf.{BigDLSessionImpl, TensorflowLoader}
import org.apache.spark.SparkContext
import scopt.OptionParser


object Utils {

  case class TrainParams(
                          trainingModelDir: String = "/tmp/tfmodel",
                          validationModelDir: Option[String] = None,
                          batchSize: Int = 16,
                          nEpochs: Int = 10)

  val trainParser = new OptionParser[TrainParams]("BigDL ptbModel Train Example") {
    opt[String]('t', "trainingModelDir")
      .text("tensorflow training model location")
      .action((x, c) => c.copy(trainingModelDir = x))
      .required()

    opt[String]('v', "validationModelDir")
      .text("tensorflow validation model location")
      .action((x, c) => c.copy(validationModelDir = Some(x)))

    opt[Int]('b', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x))

    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}

object TransferLearning {

  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {
    Utils.trainParser.parse(args, Utils.TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Transfer Learning")
        .set("spark.task.maxFailures", "1")

      val sc = new SparkContext(conf)
      Engine.init

      val trainingData = getData(param.trainingModelDir, sc)

      val model = Sequential[Float]()
      model.add(Squeeze[Float](null, batchMode = true))
      model.add(Linear[Float](1024, 5))

      val criterion = CrossEntropyCriterion[Float]()

      val optimizer = Optimizer[Float](model, trainingData, criterion, param.batchSize)

      val endWhen = Trigger.maxEpoch(param.nEpochs)
      val optim = new RMSprop[Float](learningRate = 0.001, decayRate = 0.9)

      optimizer.setEndWhen(endWhen)
      optimizer.setOptimMethod(optim)

      if (param.validationModelDir.isDefined) {
        val validationData = getData(param.validationModelDir.get, sc)
        optimizer.setValidation(Trigger.everyEpoch, validationData,
          Array(new Top1Accuracy[Float]()), param.batchSize)
      }

      optimizer.optimize()

      sc.stop()
    })
  }

  private def getData(modelDir: String, sc: SparkContext) = {
    val training_graph_file = modelDir + "/model.pb"
    val training_bin_file = modelDir + "/model.bin"

    val featureOutputNode = "InceptionV1/Logits/AvgPool_0a_7x7/AvgPool"
    val labelNode = "OneHotEncoding/one_hot"

    val session = TensorflowLoader.checkpoints[Float](training_graph_file,
      training_bin_file, ByteOrder.LITTLE_ENDIAN)
      .asInstanceOf[BigDLSessionImpl[Float]]

    val rdd = session.getRDD(Seq(featureOutputNode, labelNode), sc)

    val sampleRdd = rdd.map { t =>
      val feature = t[Tensor[Float]](1)
      val onehotLabel = t[Tensor[Float]](2)
      val label = onehotLabel.max(1)._2
      Sample(feature, label)
    }
    sampleRdd
  }

}
