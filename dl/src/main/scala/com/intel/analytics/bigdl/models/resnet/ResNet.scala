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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{MulConstant, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.Table

object ResNet {

  def shareGradInput(model: Module[Float]): Unit = {
    println("Share gradients in ResNet")
    Utils.shareGradInput(model)
  }
  def modelInit(model: Module[Float]): Unit = {
    println("Initialize ResNet")
    Utils.findModules(model)
  }

  var iChannels = 0
  def apply(classNum: Int, opt: Table): Module[Float] = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.get("dataset")
    val dataset = dataSet.getOrElse(DatasetType.CIFAR10).asInstanceOf[DatasetType]

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C || (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(SpatialBatchNormalization(nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        Sequential()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat(2)
            .add(Identity())
            .add(MulConstant(0f)))
      }  else {
        Identity()
      }
    }

    def basicBlock(n: Int, stride: Int): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n

      val s = Sequential()
      s.add(SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1))
      s.add(SpatialBatchNormalization(n))
      s.add(ReLU(true))
      s.add(SpatialConvolution(n ,n, 3, 3, 1, 1, 1, 1))
      s.add(SpatialBatchNormalization(n))

      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n, stride)))
        .add(CAddTable(true))
        .add(ReLU(true))
    }

    def bottleneck(n: Int, stride: Int): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(SpatialConvolution(nInputPlane, n, 1, 1, 1, 1, 0, 0))
        .add(SpatialBatchNormalization(n))
        .add(ReLU(true))
        .add(SpatialConvolution(n, n, 3, 3, stride, stride, 1, 1))
        .add(SpatialBatchNormalization(n))
        .add(ReLU(true))
        .add(SpatialConvolution(n, n*4, 1, 1, 1, 1, 0, 0))
        .add(SpatialBatchNormalization(n * 4))

      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n*4, stride)))
        .add(CAddTable(true))
        .add(ReLU(true))
    }

    def layer(block: (Int, Int) => Module[Float], features: Int, count: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1))
      }
      s
    }

    val model = Sequential()

    if (dataset == DatasetType.ImageNet) {
      val cfg = Map(
        18 -> ((2, 2, 2, 2), 512, basicBlock: (Int, Int) => Module[Float]),
        34 -> ((3, 4, 6, 3), 512, basicBlock: (Int, Int) => Module[Float]),
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        101 -> ((3, 4, 23, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        152 -> ((3, 8, 36, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        200 -> ((3, 24, 36, 3), 2048, bottleneck: (Int, Int) => Module[Float])
      )

      assert(cfg.keySet.contains(depth))

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      println(" | ResNet-" + depth + " ImageNet")

      //-- The ResNet ImageNet Model

      model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
        .add(SpatialBatchNormalization(64))
        .add(ReLU(true))
        .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))
        .add(layer(block, 64, loopConfig._1))
        .add(layer(block, 128, loopConfig._2, 2))
        .add(layer(block, 256, loopConfig._3, 2))
        .add(layer(block, 512, loopConfig._4, 2))
        .add(SpatialAveragePooling(7, 7, 1, 1))
        .add(View(nFeatures).setNumInputDims(3))
        .add(Linear(nFeatures, classNum))
    } else if (dataset == DatasetType.CIFAR10) {
      assert((depth-2)%6 == 0, "depth should be one of 20, 32, 44, 56, 110, 1202")
      val n = (depth-2)/6
      iChannels = 16
      println(" | ResNet-" + depth + " CIFAR-10")

      model.add(SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
      model.add(SpatialBatchNormalization(16))
      model.add(ReLU(true))
      model.add(layer(basicBlock, 16, n))
      model.add(layer(basicBlock, 32, n, 2))
      model.add(layer(basicBlock, 64, n, 2))
      model.add(SpatialAveragePooling(8, 8, 1, 1))
      model.add(View(64).setNumInputDims(3))
      model.add(Linear(64, 10))
    } else {
      sys.error("invalid dataset: " + dataset)
    }
    model
  }

  sealed abstract class DatasetType(typeId: Int)
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }
  sealed abstract class ShortcutType(typeId: Int)

  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}