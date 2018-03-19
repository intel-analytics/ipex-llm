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

package com.intel.analytics.bigdl.models.autoencoder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat

object Autoencoder {
  val rowN = 28
  val colN = 28
  val featureSize = rowN * colN

  def apply(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
    model.add(new Reshape(Array(featureSize)))
    model.add(new Linear(featureSize, classNum))
    model.add(new ReLU[Float]())
    model.add(new Linear(classNum, featureSize))
    model.add(new Sigmoid[Float]())
    model
  }

  def graph(classNum: Int): Module[Float] = {
    val input = Reshape(Array(featureSize)).inputs()
    val linear1 = Linear(featureSize, classNum).inputs(input)
    val relu = ReLU().inputs(linear1)
    val linear2 = Linear(classNum, featureSize).inputs(relu)
    val output = Sigmoid().inputs(linear2)
    Graph(input, output)
  }

  def keras(classNum: Int): nn.keras.Sequential[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val model = Sequential[Float]()
    model.add(Reshape(Array(featureSize), inputShape = Shape(28, 28)))
    model.add(Dense(classNum, activation = "relu"))
    model.add(Dense(featureSize, activation = "sigmoid"))
    model
  }

  def kerasGraph(classNum: Int): nn.keras.Model[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val input = Input(inputShape = Shape(28, 28))
    val reshape = Reshape(Array(featureSize)).inputs(input)
    val dense1 = Dense(classNum, activation = "relu").inputs(reshape)
    val output = Dense(featureSize, activation = "sigmoid").inputs(dense1)
    Model[Float](input, output)
  }
}
