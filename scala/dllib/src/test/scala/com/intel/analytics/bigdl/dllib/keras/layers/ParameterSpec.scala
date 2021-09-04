/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.{MSECriterion, Module}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.Sequential
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Lambda, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.math.log
import scala.reflect.ClassTag


class ParameterSpec extends KerasBaseSpec {

  "Parameter add" should "be test" in {
    val w = Parameter[Float](Shape(2, 3))
    val x = Variable[Float](Shape(2, 3))
    val c = x + w
  }

  "sub" should "be test" in {
    val x = Variable[Float](Shape(1))
    val y = Variable[Float](Shape(1))
    val c = x + 1 - y
    val model = Model[Float](input = Array(x, y), output = c)
    model.forward(T(Tensor[Float](Array(4, 1)).rand(0, 1), Tensor[Float](Array(4, 1)).rand(0, 1)))

  }

  "Parameter matMul axes" should "be test" in {
    val w = Parameter[Float](Shape(3, 2))
    val x = Variable[Float](Shape(3))
    val bias = Parameter[Float](Shape(2))
    val c = AutoGrad.mm(w, x, axes = List(0, 1)) + bias
  }

  "Parameter matMul" should "be test" in {
    val w = Parameter[Float](Shape(2, 3))
    val initWeight = w.getWeight()
    val x = Variable[Float](Shape(3))
    val bias = Parameter[Float](Shape(2), initWeight =
      Tensor[Float](Shape(2).toSingle().toArray).rand(0, 1))
    val c = AutoGrad.mm(x, w, axes = List(1, 1)) + bias
    w.setWeight(Tensor[Float](Shape(2, 3).toSingle().toArray).rand(0, 1))
  }


  "Use parameter to construct a linear layer" should "be test" in {
    def cDense(): Model[Float] = {
      val input = Variable[Float](Shape(3))
      val w = Parameter[Float](Shape(2, 3)) // outputSize * inputSize
      val bias = Parameter[Float](Shape(2))
      val cDense = AutoGrad.mm(input, w, axes = List(1, 1)) + bias
      val model = Model[Float](input = input, output = cDense)
      model
    }

    def train(i: Int, input: Tensor[Float], res: Tensor[Float],
        module: AbstractModule[Activity, Activity, Float],
        loss: AbstractCriterion[Activity, Activity, Float]): Unit = {
      val output = module.forward(input)
      loss.forward(output, res)
      val grad = loss.backward(output, res)
      module.zeroGradParameters()
      module.backward(input, grad)
      val (weight, gradWeight) = KerasUtils.invokeMethod(
        module, "getParameters").asInstanceOf[(Tensor[Float], Tensor[Float])]
      weight.add((-0.5f / log(i + 3)).toFloat, gradWeight)
    }

    val inputData = Tensor[Float](Shape(4, 3).toSingle().toArray).rand(0, 1).fill(1f)

    val seq = Sequential[Float]()
    seq.add(Dense[Float](outputDim = 2, inputShape = Shape(1, 3), bias = true))
    val cmodel = cDense()
    cmodel.setWeightsBias(seq.getWeightsBias().reverse)


    val batchN = 80
    val inputN = 3
    val outputN = 2
    val input = Tensor[Float](batchN, inputN)
    val res = Tensor[Float](batchN, outputN)
    val mse = new MSECriterion[Float]

    var err = 0.0
    for (i <- 1 to 1000) {
      input.rand()
      for (k <- 1 to batchN) {
        for (y <- 1 to outputN) {
          res(Array(k, y)) = 1.0f * y
          for (x <- 1 to inputN) {
            res(Array(k, y)) += 0.1f * y * x * input(Array(k, x))
          }
        }
      }
      train(i, input, res, seq, mse)
      train(i, input, res, cmodel, mse)
    }
    assert(seq.getWeightsBias()(0).almostEqual(cmodel.getWeightsBias()(1), 1e-4))
    assert(seq.getWeightsBias()(1).almostEqual(cmodel.getWeightsBias()(0), 1e-4))
  }

  "Parameter save/load" should "be able to work" in {
    val w = Parameter[Float](Shape(1, 10),
      initWeight = Tensor.ones[Float](10).view(1, 10))
    val b = Parameter[Float](Shape(1, 10),
      initWeight = Tensor[Float](10).view(1, 10))
    val x = Variable[Float](Shape(1, 10))
    val z = x * w
    val model = Model[Float](input = x, output = z)
    val input2 = Tensor[Float](Array(1, 10)).rand
    val out = model.forward(input2.clone()).toTensor[Float].clone()
    val gradOutput = Tensor[Float](out.size())
    val gradInput = model.backward(input2, gradOutput.clone()).toTensor[Float].clone()

    val tmpFile = ZooSpecHelper.createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    model.saveModule(absPath, overWrite = true)
    val model2 = Module.loadModule[Float](absPath)
    val output2 = model2.forward(input2).toTensor[Float]
    val gradInput2 = model2.backward(input2, gradOutput).toTensor[Float]
    require(out.almostEqual(output2, 1e-8))
    require(gradInput.almostEqual(gradInput2, 1e-9))
  }
}
