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

package com.intel.analytics.bigdl.models.resnet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import Options._

object Train {

    def main(args: Array[String]): Unit = {
      trainLocalParser.parse(args, new TrainLocalParams()).map(param => {

        val batchSize = 64
        val (imageSize, lrSchedule, epochEnd, dataSet) = param.dataset match {
          case "imagenet" => (224, DatasetType.ImageNet, 90, ImagenetDataSet)
          case _ => (32, DatasetType.CIFAR10, 165, Cifar10DataSet)
        }

        val trainData = Paths.get(param.folder, "train")
        val trainDataSet = dataSet.localTrainDataSet(trainData, true, batchSize)
        val validationData = Paths.get(param.folder, "val")
        val validateDataSet = dataSet.localValDataSet(validationData, true, batchSize)

        val model = if (param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          val curModel = param.dataset match {
            case "imagenet" => ResNet(classNum = 100, T("shortcutType" -> ShortcutType.B, "depth" -> 18))
            case _ => ResNet(classNum = 10, T("shortcutType" -> ShortcutType.A, "depth" -> 20))
          }
          ResNet.modelInit(curModel)
          curModel
        }

        if (param.optnet) {
          ResNet.shareGradInput(model)
        }

        val state = if (param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> 0.1,
            "weightDecay" -> 1e-4,
            "momentum" -> 0.9,
            "dampening" -> 0.9,
            "learningRateSchedule" -> SGD.EpochDecay(lrSchedule)
          )
        }

        Engine.setCoreNumber(param.coreNumber)
        val optimizer = new LocalOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new CrossEntropyCriterion[Float]()
        )
        if (param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.severalIteration(3910))
        }
        optimizer
          .setState(state)
          .setValidation(Trigger.severalIteration(391),
            validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
          .setEndWhen(Trigger.maxEpoch(epochEnd))
          .optimize()
      })
    }

}
