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

package com.intel.analytics.bigdl.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor

object MNIST {
  val rowN = 28
  val colN = 28
  val featureSize = rowN * colN
  val classNum = 10

  def toTensor(mean: Double, std: Double)(inputs: Seq[Array[Byte]], input: Tensor[Double],
    target: Tensor[Double]): (Tensor[Double], Tensor[Double]) = {
    val size = inputs.size
    input.resize(Array(size, rowN, colN))
    target.resize(Array(size))
    var i = 0
    while (i < size) {
      val img = inputs(i)
      var j = 0
      while (j < featureSize) {
        input.setValue(i + 1, j / rowN + 1, j % rowN + 1,
          ((img(j + 1) & 0xff) / 255.0 - mean) / std)
        j += 1
      }
      target.setValue(i + 1, (img(0) & 0xff) + 1.0)
      i += 1
    }
    (input, target)
  }

  def getModule(netType: String)(): Module[Double] = {
    netType.toLowerCase match {
      case "ann" =>
        val mlp = Sequential[Double]
        val nhiddens = featureSize / 2
        mlp.add(Reshape(Array(featureSize)))
        mlp.add(Linear(featureSize, nhiddens))
        mlp.add(Tanh())
        mlp.add(Linear(nhiddens, classNum))
        mlp.add(LogSoftMax())
        mlp
      case "linear" =>
        val mlp = Sequential[Double]
        mlp.add(Reshape(Array(featureSize)))
        mlp.add(Linear(featureSize, classNum))
        mlp.add(LogSoftMax())
        mlp
      case "cnn" =>
        val model = Sequential[Double]()
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
      case "lenet" =>
        val model = Sequential[Double]()
        model.add(Reshape(Array(1, rowN, colN)))
        model.add(SpatialConvolution(1, 6, 5, 5))
        model.add(Tanh())
        model.add(SpatialMaxPooling(2, 2, 2, 2))
        model.add(Tanh())
        model.add(SpatialConvolution(6, 12, 5, 5))
        model.add(SpatialMaxPooling(2, 2, 2, 2))

        model.add(Reshape(Array(12 * 4 * 4)))
        model.add(Linear(12 * 4 * 4, 100))
        model.add(Tanh())
        model.add(Linear(100, 10))
        model.add(LogSoftMax())
        model
      case _ =>
        throw new UnsupportedOperationException
    }
  }

  def loadFile(featureFile: String, labelFile: String): Array[Array[Byte]] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelMagicNumber = labelBuffer.getInt()
    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)
    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)
    val rowNum = featureBuffer.getInt()
    require(rowNum == rowN)
    val colNum = featureBuffer.getInt()
    require(colNum == colN)
    val result = new Array[Array[Byte]](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum + 1))
      img(0) = labelBuffer.get()
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(1 + x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = img
      i += 1
    }

    result
  }
}
