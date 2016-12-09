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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn.MulConstant
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Activities, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.collection.mutable
import scala.reflect.ClassTag


object ResNet {
  def shareGradInput[@specialized(Float, Double) T: ClassTag](model: Module[Tensor[T], Tensor[T], T])
                                                             (implicit ev: TensorNumeric[T]): Unit = {
    def sharingKey(m: Module[_ <: Activities, _ <: Activities, T]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[T]]()
    val packageName: String = model.getName().stripSuffix("Sequential")

    cache.put("fInput", Storage(Array(ev.fromType[Int](1))))
    cache.put("fGradInput", Storage(Array(ev.fromType[Int](1))))

    var index = 0
    def matchModels(model: Module[_ <: Activities, _ <: Activities, T]): Unit = {
      model match {
        case container: Container[Activities, Activities, T]
          => {
          container.modules.foreach( m => {

            if (m.gradInput.isInstanceOf[Tensor[T]] && !m.getClass.getName.equals(packageName + "ConcatTable")) {
              val key = sharingKey(m)
              if (!cache.contains(key)){
                cache.put(key, Storage(Array(ev.fromType[Int](1))))
              }
              val tmpModel = m.asInstanceOf[Module[Tensor[T], Tensor[T], T]]
              tmpModel.gradInput = Tensor[T](cache.get(key).get, 1, Array(0))
            }
            if (m.getClass.getName.equals(packageName + "ConcatTable")) {
              if (!cache.contains(index % 2)) cache.put(index%2, Storage(Array(ev.fromType[Int](1))))
              val tmpModel = m.asInstanceOf[ConcatTable[Tensor[T], T]]
              tmpModel.gradInput = Tensor[T](cache.get(index % 2).get, 1, Array(0))
              index = index + 1
            }
            matchModels(m)
          })
        }
        case spatialConvolution: SpatialConvolution[T]
          => {
          spatialConvolution.setSharedVar
          spatialConvolution.fInput = Tensor[T](cache.get("fInput").get)
          spatialConvolution.fGradInput = Tensor[T](cache.get("fGradInput").get)
        }
        case _ => Unit
      }
    }

    matchModels(model)
  }

  def findModules[@specialized(Float, Double) T: ClassTag](model: Module[_ <: Activities, _ <: Activities, T])
                                                          (implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container: Container[Activities, Activities, T]
      => container.modules.foreach(m => findModules(m))
      case spatialConvolution: SpatialConvolution[T]
      => {
        val n: Float = spatialConvolution.kernelW * spatialConvolution.kernelW * spatialConvolution.nOutputPlane
        spatialConvolution.weight.apply1(_ => ev.fromType[Float](RNG.normal(0, Math.sqrt(2.0f / n)).toFloat))
        spatialConvolution.bias.apply1(_ => ev.fromType[Float](0))
      }
      case spatialBatchNormalization: SpatialBatchNormalization[T]
      => {
        spatialBatchNormalization.weight.apply1(_ => ev.fromType[Float](1.0f))
        spatialBatchNormalization.bias.apply1(_ => ev.fromType[Float](0.0f))
      }
      case linear: Linear[T]
      => {
        linear.bias.apply1(_ => ev.fromType[Float](0))
      }
      case _ => Unit
    }
  }

  def modelInit[@specialized(Float, Double) T: ClassTag](model: Module[_ <:Activities, _ <:Activities, T])
                                                        (implicit ev: TensorNumeric[T]): Unit = {
    findModules(model)
  }

  var iChannels = 0
  def apply[T: ClassTag](classNum: Int, opt: Table)(implicit ev: TensorNumeric[T]): Module[Activities, Activities, T] = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.get("dataset")
    val dataset = dataSet.getOrElse(DatasetType.CIFAR10).asInstanceOf[DatasetType]

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Activities, Activities, T] = {
      val useConv = shortcutType == ShortcutType.C || (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        new Sequential[Activities, Activities, T]()
          .add(new SpatialConvolution[T](nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(new SpatialBatchNormalization(nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        new Sequential[Activities, Activities, T]()
          .add(new SpatialAveragePooling[T](1, 1, stride, stride))
          .add(new Concat[T](2)
            .add(new Identity[T]())
            .add(new MulConstant[T](ev.fromType(0))))
      }  else {
        new Identity()
      }
    }

    def basicBlock(n: Int, stride: Int): Module[Activities, Activities, T] = {
      val nInputPlane = iChannels
      iChannels = n

      val s = new Sequential[Activities, Activities, T]()
      s.add(new SpatialConvolution[T](nInputPlane, n, 3, 3, stride, stride, 1, 1))
      s.add(new SpatialBatchNormalization[T](n))
      s.add(new ReLU[T](true))
      s.add(new SpatialConvolution[T](n ,n, 3, 3, 1, 1, 1, 1))
      s.add(new SpatialBatchNormalization[T](n))

      new Sequential[Activities, Activities, T]()
        .add(new ConcatTable[Tensor[T], T]()
          .add(s)
          .add(shortcut(nInputPlane, n, stride)))
        .add(new CAddTable(true))

        .add(new ReLU[T](true))
    }

    def bottleneck(n: Int, stride: Int): Module[Activities, Activities, T] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = new Sequential[Activities, Activities, T]()
      s.add(new SpatialConvolution[T](nInputPlane, n, 1, 1, 1, 1, 0, 0))
        .add(new SpatialBatchNormalization[T](n))
        .add(new ReLU[T](true))
        .add(new SpatialConvolution[T](n, n, 3, 3, stride, stride, 1, 1))
        .add(new SpatialBatchNormalization[T](n))
        .add(new ReLU[T](true))
        .add(new SpatialConvolution[T](n, n*4, 1, 1, 1, 1, 0, 0))
        .add(new SpatialBatchNormalization[T](n * 4))

      new Sequential[Activities, Activities, T]()
        .add(new ConcatTable[Tensor[T], T]()
          .add(s)
          .add(shortcut(nInputPlane, n*4, stride)))
        .add(new CAddTable(true))
        .add(new ReLU[T](true))
    }

    def layer(block: (Int, Int) => Module[Activities, Activities, T], features: Int, count: Int, stride: Int = 1): Module[Activities, Activities, T] = {
      val s = new Sequential[Activities, Activities, T]()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1))
      }
      s
    }

    val model = new Sequential[Activities, Activities, T]()

    if (dataset == DatasetType.ImageNet) {
      val cfg = Map(
        18 -> ((2, 2, 2, 2), 512, basicBlock: (Int, Int) => Module[Activities, Activities, T]),
        34 -> ((3, 4, 6, 3), 512, basicBlock: (Int, Int) => Module[Activities, Activities, T]),
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int) => Module[Activities, Activities, T]),
        101 -> ((3, 4, 23, 3), 2048, bottleneck: (Int, Int) => Module[Activities, Activities, T]),
        152 -> ((3, 8, 36, 3), 2048, bottleneck: (Int, Int) => Module[Activities, Activities, T]),
        200 -> ((3, 24, 36, 3), 2048, bottleneck: (Int, Int) => Module[Activities, Activities, T])
      )

      assert(cfg.keySet.contains(depth))

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      println(" | ResNet-" + depth + " ImageNet")

      //-- The ResNet ImageNet Model

      model.add(new SpatialConvolution[T](3, 64, 7, 7, 2, 2, 3, 3))
        .add(new SpatialBatchNormalization[T](64))
        .add(new ReLU[T](true))
        .add(new SpatialMaxPooling[T](3, 3, 2, 2, 1, 1))
        .add(layer(block, 64, loopConfig._1))
        .add(layer(block, 128, loopConfig._2, 2))
        .add(layer(block, 256, loopConfig._3, 2))
        .add(layer(block, 512, loopConfig._4, 2))
        .add(new SpatialAveragePooling[T](7, 7, 1, 1))
        .add(new View[T](nFeatures).setNumInputDims(3))
        .add(new Linear[T](nFeatures, classNum))
    } else if (dataset == DatasetType.CIFAR10) {
      assert((depth-2)%6 == 0, "depth should be one of 20, 32, 44, 56, 110, 1202")
      val n = (depth-2)/6
      iChannels = 16
      println(" | ResNet-" + depth + " CIFAR-10")

      model.add(new SpatialConvolution[T](3, 16, 3, 3, 1, 1, 1, 1))
      model.add(new SpatialBatchNormalization[T](16))
      model.add(new ReLU[T](true))
      model.add(layer(basicBlock, 16, n))
      model.add(layer(basicBlock, 32, n, 2))
      model.add(layer(basicBlock, 64, n, 2))
      model.add(new SpatialAveragePooling[T](8, 8, 1, 1))
      model.add(new View[T](64).setNumInputDims(3))
      model.add(new Linear[T](64, 10))
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