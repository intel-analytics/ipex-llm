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

package com.intel.analytics.sparkdl.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor

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
        val mlp = new Sequential[Double]
        val nhiddens = featureSize / 2
        mlp.add(new Reshape(Array(featureSize)))
        mlp.add(new Linear(featureSize, nhiddens))
        mlp.add(new Tanh)
        mlp.add(new Linear(nhiddens, classNum))
        mlp.add(new LogSoftMax)
        mlp
      case "linear" =>
        val mlp = new Sequential[Double]
        mlp.add(new Reshape(Array(featureSize)))
        mlp.add(new Linear(featureSize, classNum))
        mlp.add(new LogSoftMax)
        mlp
      case "cnn" =>
        val model = new Sequential[Double]()
        model.add(new Reshape(Array(1, rowN, colN)))
        model.add(new SpatialConvolution(1, 32, 5, 5))
        model.add(new Tanh())
        model.add(new SpatialMaxPooling(3, 3, 3, 3))
        model.add(new SpatialConvolution(32, 64, 5, 5))
        model.add(new Tanh())
        model.add(new SpatialMaxPooling(2, 2, 2, 2))

        val linearInputNum = 64 * 2 * 2
        val hiddenNum = 200
        model.add(new Reshape(Array(linearInputNum)))
        model.add(new Linear(linearInputNum, hiddenNum))
        model.add(new Tanh())
        model.add(new Linear(hiddenNum, classNum))
        model.add(new LogSoftMax())
        model
      case "lenet" =>
        val model = new Sequential[Double]()
        model.add(new Reshape(Array(1, rowN, colN)))
        model.add(new SpatialConvolution(1, 6, 5, 5))
        model.add(new Tanh())
        model.add(new SpatialMaxPooling(2, 2, 2, 2))
        model.add(new Tanh())
        model.add(new SpatialConvolution(6, 12, 5, 5))
        model.add(new SpatialMaxPooling(2, 2, 2, 2))

        model.add(new Reshape(Array(12 * 4 * 4)))
        model.add(new Linear(12 * 4 * 4, 100))
        model.add(new Tanh())
        model.add(new Linear(100, 10))
        model.add(new LogSoftMax())
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
