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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.reflect.ClassTag


object Convolution {
  def apply[@specialized(Float, Double) T: ClassTag](
     nInputPlane: Int,
     nOutputPlane: Int,
     kernelW: Int,
     kernelH: Int,
     strideW: Int = 1,
     strideH: Int = 1,
     padW: Int = 0,
     padH: Int = 0,
     nGroup: Int = 1,
     propagateBack: Boolean = true,
     initMethod: InitializationMethod = Default,
     optnet: Boolean = true)
     (implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    if (optnet) {
      SpatialShareConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack, initMethod)
    } else {
      SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack, initMethod)
    }
  }
}

object ResNet {
  val logger = Logger.getLogger(getClass)

  def shareGradInput(model: Module[Float]): Unit = {
    logger.info("Share gradients in ResNet")
    def sharingKey(m: Module[Float]) = m.getClass.getName
    val cache = mutable.Map[Any, Storage[Float]]()
    val packageName: String = model.getName().stripSuffix("Sequential")
    cache.put("fInput", Storage(Array(1.0f)))
    cache.put("fGradInput", Storage(Array(1.0f)))

    var index = 0
    def matchModels(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float] =>
          container.modules.foreach( m => {
            if (m.gradInput.isInstanceOf[Tensor[_]] &&
              !m.getClass.getName.equals(packageName + "ConcatTable")) {
              val key = sharingKey(m)
              if (!cache.contains(key)) {
                cache.put(key, Storage(Array(1.0f)))
              }
              m.gradInput = Tensor(cache.get(key).get, 1, Array(0))
            }
            matchModels(m)
          })
        case concatTable if (concatTable.isInstanceOf[ConcatTable[Float]]) =>
          if (!cache.contains(index % 2)) {
            cache.put(index % 2, Storage(Array(1.0f)))
          }
          concatTable.gradInput = Tensor[Float](cache.get(index % 2).get, 1, Array(0))
          index = index + 1
        case spatialShareConvolution
          if (spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Float]]) =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Float]]
          curModel.fInput = Tensor[Float](cache.get("fInput").get)
          curModel.fGradInput = Tensor[Float](cache.get("fGradInput").get)
        case _ => Unit
      }
    }
    matchModels(model)
  }

  def modelInit(model: Module[Float]): Unit = {
    logger.info("Initialize ResNet")
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float]
        => container.modules.foreach(m => initModules(m))
        case spatialShareConvolution
          if (spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Float]]) =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Float]]
          val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
        case spatialConvolution
          if (spatialConvolution.isInstanceOf[SpatialConvolution[Float]]) =>
          val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[Float]]
          val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
        case spatialBatchNormalization
          if (spatialBatchNormalization.isInstanceOf[SpatialBatchNormalization[Float]]) =>
          val curModel = spatialBatchNormalization.asInstanceOf[SpatialBatchNormalization[Float]]
          curModel.weight.apply1(_ => 1.0f)
          curModel.bias.apply1(_ => 0.0f)
        case linear if (linear.isInstanceOf[Linear[Float]]) =>
          linear.asInstanceOf[Linear[Float]].bias.apply1(_ => 0.0f)
        case _ => Unit
      }
    }
    initModules(model)
  }

  var iChannels = 0
  def apply(classNum: Int, opt: Table): Module[Float] = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.get("dataset")
    val dataset = dataSet.getOrElse(DatasetType.CIFAR10).asInstanceOf[DatasetType]
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet))
          .add(SpatialBatchNormalization(nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        Sequential()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat(2)
            .add(Identity())
            .add(MulConstant(0f)))
      } else {
        Identity()
      }
    }

    def basicBlock(n: Int, stride: Int): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n

      val s = Sequential()
      s.add(Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
      s.add(SpatialBatchNormalization(n))
      s.add(ReLU(true))
      s.add(Convolution(n, n, 3, 3, 1, 1, 1, 1, optnet = optnet))
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
      s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet))
        .add(SpatialBatchNormalization(n))
        .add(ReLU(true))
        .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
        .add(SpatialBatchNormalization(n))
        .add(ReLU(true))
        .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet))
        .add(SpatialBatchNormalization(n * 4))

      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n*4, stride)))
        .add(CAddTable(true))
        .add(ReLU(true))
    }

    def layer(block: (Int, Int) => Module[Float], features: Int,
              count: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1))
      }
      s
    }

    val model = Sequential()
    if (dataset == DatasetType.ImageNet) {
      val cfg = Map(
        18 -> ((2, 2, 2, 2), 512,
          basicBlock: (Int, Int) => Module[Float]),
        34 -> ((3, 4, 6, 3), 512,
          basicBlock: (Int, Int) => Module[Float]),
        50 -> ((3, 4, 6, 3), 2048,
          bottleneck: (Int, Int) => Module[Float]),
        101 -> ((3, 4, 23, 3), 2048,
          bottleneck: (Int, Int) => Module[Float]),
        152 -> ((3, 8, 36, 3), 2048,
          bottleneck: (Int, Int) => Module[Float]),
        200 -> ((3, 24, 36, 3), 2048,
          bottleneck: (Int, Int) => Module[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      logger.info(" | ResNet-" + depth + " ImageNet")

      model.add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = optnet))
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
      require((depth - 2)%6 == 0,
        "depth should be one of 20, 32, 44, 56, 110, 1202")
      val n = (depth-2)/6
      iChannels = 16
      logger.info(" | ResNet-" + depth + " CIFAR-10")

      model.add(Convolution(3, 16, 3, 3, 1, 1, 1, 1, optnet = optnet))
      model.add(SpatialBatchNormalization(16))
      model.add(ReLU(true))
      model.add(layer(basicBlock, 16, n))
      model.add(layer(basicBlock, 32, n, 2))
      model.add(layer(basicBlock, 64, n, 2))
      model.add(SpatialAveragePooling(8, 8, 1, 1))
      model.add(View(64).setNumInputDims(3))
      model.add(Linear(64, 10))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataset}")
    }
    model
  }

  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}
