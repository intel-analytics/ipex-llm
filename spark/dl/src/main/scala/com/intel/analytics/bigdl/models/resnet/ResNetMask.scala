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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.reflect.ClassTag

object ResNetMask {
  val logger = Logger.getLogger(getClass)

  var iChannels = 0
  var lastChannel = 0

  def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
               useConv: Boolean = false): Module[Float] = {
    //      val useConv = shortcutType == ShortcutType.C ||
    //        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

    if (useConv) {
      Sequential()
        .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        .add(Sbn(nOutputPlane))
    } else {
      Identity()
    }
  }


  def creatLayer1(): Module[Float] = {
    val number = 2

    val s = Sequential()
    s.add(Convolution(64, 64, 1, 1, 1, 1, 0, 0))
      .add(Sbn(64))
      .add(ReLU(true))
      .add(Convolution(64, 64, 3, 3, 1, 1, 1, 1))
      .add(Sbn(64))
      .add(ReLU(true))
      .add(Convolution(64, 32, 1, 1, 1, 1, 0, 0))
      .add(Sbn(32))
    val m1 = Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(64, 32, 1, true)))
      .add(CAddTable(true))
      .add(ReLU(true))

    val model = Sequential().add(m1)

    for (i <- 0 to number - 1) {
      val m2 = Sequential()
        .add(Convolution(32, 64, 1, 1, 1, 1, 0, 0))
        .add(Sbn(64))
        .add(ReLU(true))
        .add(Convolution(64, 64, 3, 3, 1, 1, 1, 1))
        .add(Sbn(64))
        .add(ReLU(true))
        .add(Convolution(64, 32, 1, 1, 1, 1, 0, 0))
        .add(Sbn(32))

      val short = Sequential()
        .add(ConcatTable()
          .add(m2)
          .add(shortcut(-1, -1, -1, false)))
        .add(CAddTable(true))
        .add(ReLU(true))

      model.add(short)
    }
    model
  }

  def creatLayer2(): Module[Float] = {
    val n1 = 32
    val n2 = 128
    val n3 = 64

    val number = 3

    val s = Sequential()
    s.add(Convolution(n1, n2, 1, 1, 2, 2, 0, 0))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
      .add(Sbn(n3))
    val m1 = Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(n1, n3, 2, true)))
      .add(CAddTable(true))
      .add(ReLU(true))

    val model = Sequential().add(m1)

    for (i <- 0 to number - 1) {
      val m2 = Sequential()
        .add(Convolution(n3, n2, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n3))

      val short = Sequential()
        .add(ConcatTable()
          .add(m2)
          .add(shortcut(-1, -1, -1, false)))
        .add(CAddTable(true))
        .add(ReLU(true))

      model.add(short)
    }
    model
  }

  def creatLayer3(): Module[Float] = {
    val n1 = 64
    val n2 = 256
    val n3 = 128

    val number = 5

    val s = Sequential()
    s.add(Convolution(n1, n2, 1, 1, 2, 2, 0, 0, optnet = false))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
      .add(Sbn(n3))
    val m1 = Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(n1, n3, 2, true)))
      .add(CAddTable(true))
      .add(ReLU(true))

    val model = Sequential().add(m1)

    for (i <- 0 to number - 1) {
      val m2 = Sequential()
        .add(Convolution(n3, n2, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n3))

      val short = Sequential()
        .add(ConcatTable()
          .add(m2)
          .add(shortcut(-1, -1, -1, false)))
        .add(CAddTable(true))
        .add(ReLU(true))

      model.add(short)
    }
    model
  }

  def creatLayer4(): Module[Float] = {
    val n1 = 128
    val n2 = 512
    val n3 = 256

    val number = 2

    val s = Sequential()
    s.add(Convolution(n1, n2, 1, 1, 2, 2, 0, 0, optnet = false))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
      .add(Sbn(n2))
      .add(ReLU(true))
      .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
      .add(Sbn(n3))
    val m1 = Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(n1, n3, 2, true)))
      .add(CAddTable(true))
      .add(ReLU(true))

    val model = Sequential().add(m1)

    for (i <- 0 to number - 1) {
      val m2 = Sequential()
        .add(Convolution(n3, n2, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n2, 3, 3, 1, 1, 1, 1, optnet = false))
        .add(Sbn(n2))
        .add(ReLU(true))
        .add(Convolution(n2, n3, 1, 1, 1, 1, 0, 0, optnet = false))
        .add(Sbn(n3))

      val short = Sequential()
        .add(ConcatTable()
          .add(m2)
          .add(shortcut(-1, -1, -1, false)))
        .add(CAddTable(true))
        .add(ReLU(true))

      model.add(short)
    }
    model
  }

  def apply(classNum: Int, opt: Table): Module[Float] = {
    val depth = opt.get("depth").getOrElse(18)
    val optnet = false


    val model = Sequential()
    if (true) {
      iChannels = 32
      lastChannel = 32
      logger.info(" | ResNet-" + depth + " ImageNet")

      model.add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = optnet, propagateBack = false))
        .add(Sbn(64))
        .add(ReLU(true))
        .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

//      val layer1 = creatLayer1()
//      val layer2 = creatLayer2()
//      val layer3 = creatLayer3()
//      val layer4 = creatLayer4()
//      model.add(layer1)

      return model
    } else {
      throw new IllegalArgumentException(s"Invalid dataset")
    }
  }

  /**
   * dataset type
   * @param typeId type id
   */
  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  /**
   *  define some dataset type
   */
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }

  /**
   * ShortcutType
   * @param typeId type id
   */
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  /**
   * ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
   * ShortcutType-C is used for others.
   */
  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}
