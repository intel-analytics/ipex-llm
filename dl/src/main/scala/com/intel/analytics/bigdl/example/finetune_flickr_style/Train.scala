/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.example.finetune_flickr_style

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext

object Train {

  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = Engine.init(param.nodesNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL Flickr Style fine tune Example")
            .set("spark.task.maxFailures", "1").setMaster("local")
          new SparkContext(conf)
        })
      val imageSize = 227
      // create dataset
      val trainData = Paths.get(param.folder, "train")
      val trainDataSet = FlickrImage.load(trainData, sc, imageSize, param.batchSize)
      val validationData = Paths.get(param.folder, "test")
      val validateDataSet = FlickrImage.load(validationData, sc, imageSize, param.batchSize)
      // Create model
      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else if (param.modelPath != "") {
        param.modelName match {
          case "CaffeNet" =>
            Module.loadCaffe[Float](CaffeNetFlickr(20),
              param.caffeDefPath, param.modelPath, false)
          case "GoogleNet" =>
            Module.loadCaffe[Float](Flickr_Style_Googlenet(20),
              param.caffeDefPath, param.modelPath, false)
          case _ => throw new IllegalArgumentException(s"${param.modelName}")
        }
      } else {
        param.modelName match {
          case "CaffeNet" =>
            CaffeNetFlickr(20)
          case "GoogleNet" =>
            Flickr_Style_Googlenet(classNum = 20)
          case _ => throw new IllegalArgumentException(s"${param.modelName}")
        }

      }
      // set hyper paramters
      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        param.modelName match {
          case "CaffeNet" =>
            T(
              "learningRate" -> param.learningRate,
              "weightDecay" -> 0.0005,
              "momentum" -> 0.9,
              "learningRateSchedule" -> SGD.EpochStep(12, 0.1)
            )
          case "GoogleNet" =>
            T(
              "learningRate" -> param.learningRate,
              "weightDecay" -> 0.0002,
              "momentum" -> 0.9,
              "learningRateSchedule" -> SGD.Poly(0.5,
                param.maxEpoch * trainDataSet.size().toInt / param.batchSize)
            )
          case _ => throw new IllegalArgumentException(s"${param.modelName}")
        }
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new ClassNLLCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.severalIteration(1000))
      }
      optimizer
        .setState(state)
        .setValidation(Trigger.everyEpoch,
          validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()

    })
  }
}
