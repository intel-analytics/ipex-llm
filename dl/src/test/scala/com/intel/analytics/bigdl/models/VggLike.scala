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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{LogSoftMax, _}
import com.intel.analytics.bigdl.numeric.NumericDouble

object VggLike {
  def apply(classNum: Int): Module[Double] = {
    val vggBnDo = Sequential[Double]()
    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[Double] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(SpatialBatchNormalization[Double](nOutPutPlane, 1e-3).setInit())
      vggBnDo.add(ReLU(true))
      vggBnDo
    }
    convBNReLU(3, 64) // .add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling[Double](2, 2, 2, 2).ceil())

    convBNReLU(64, 128) // .add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling[Double](2, 2, 2, 2).ceil())

    convBNReLU(128, 256)// .add(Dropout(0.4))
    convBNReLU(256, 256) // .add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling[Double](2, 2, 2, 2).ceil())

    convBNReLU(256, 512) // .add(Dropout(0.4))
    convBNReLU(512, 512) // .add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling[Double](2, 2, 2, 2).ceil())

    convBNReLU(512, 512) // .add(Dropout(0.4))
    convBNReLU(512, 512) // .add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling[Double](2, 2, 2, 2).ceil())
    vggBnDo.add(View(512))

    val classifier = Sequential[Double]()
    // classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    classifier.add(BatchNormalization[Double](512).setInit())
    classifier.add(ReLU(true))
    // classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }
}

object LeNet5 {
  def apply(classNum: Int): Module[Double] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
    model.add(SpatialConvolution(1, 6, 5, 5))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Tanh())
    model.add(SpatialConvolution(6, 12, 5, 5))
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Reshape(Array(12 * 4 * 4)))
    model.add(Linear(12 * 4 * 4, 100))
    model.add(Tanh())
    model.add(Linear(100, classNum))
    model.add(LogSoftMax())
    model
  }
}

object SimpleCNN {
  val rowN = 28
  val colN = 28
  val featureSize = rowN * colN

  def apply(classNum: Int): Module[Double] = {
    val model = Sequential()
    model.add(Reshape(Array(1, rowN, colN)))
    model.add(SpatialConvolution(1, 32, 5, 5))
    model.add(Tanh())
    model.add(SpatialMaxPooling(3, 3, 3, 3))
    model.add(SpatialConvolution(32, 64, 5, 5))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    val linearInputNum = 64 * 2 * 2
    val hiddenNum = 200
    model.add(Reshape(Array(linearInputNum)))
    model.add(Linear(linearInputNum, hiddenNum))
    model.add(Tanh())
    model.add(Linear(hiddenNum, classNum))
    model.add(LogSoftMax())
    model
  }
}
