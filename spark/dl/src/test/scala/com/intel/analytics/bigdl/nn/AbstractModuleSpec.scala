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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.util.Random

class AbstractModuleSpec extends FlatSpec with Matchers {
  "Get name" should "find the module if it exists" in {
    val m = Linear(4, 3).setName("module")
    m("module").get should be(m)
  }

  "Get name" should "find the module if it exists in container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)

    s("module").get should be(m)
  }

  "Get name" should "find the module if it exists in deeper container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)
    val k = Sequential()
    k.add(s)

    k("module").get should be(m)
  }

  "Get name" should "get the container if it is the container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.setName("container")
    s.add(m)

    s("container").get should be(s)
  }

  "Get name" should "not find if there is no such module" in {
    val m = Linear(4, 3)
    m("module") should be(None)
    val s = Sequential()
    s.add(m)
    s("container") should be(None)
  }

  "Get name" should "throw exception if there are two modules with same name" in {
    val m1 = Linear(4, 3)
    val m2 = Linear(4, 3)
    m1.setName("module")
    m2.setName("module")
    val s = Sequential()
    s.add(m1).add(m2)

    intercept[IllegalArgumentException] {
      s("module").get
    }
  }

  "weights save and load" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(Tanh())
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(Tanh())
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Tanh())
    module2.add(Linear(100, 6).setName("l2"))
    module2.add(LogSoftMax())

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "weights save and load with different model definition" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Linear(100, 6).setName("l2"))

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "weights save and load with only weight or bias" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(CMul(Array(1, 4, 1, 1)).setName("cmul"))
    module.add(CAdd(Array(1, 4, 1, 1)).setName("cadd"))

    val module2 = Sequential()

    module2.add(CMul(Array(1, 4, 1, 1)).setName("cmul"))
    module2.add(CAdd(Array(1, 4, 1, 1)).setName("cadd"))

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "loadModelWeights" should "work properly" in {
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Linear(100, 6).setName("l2"))
    module.loadModelWeights(module2)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "freeze and unfreeze" should "work properly" in {
    def inceptionLayerV1Seq(inputSize: Int, config: Table) : Module[Float] = {
      val concat = Concat(2)
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName("conv1x1").setScaleW(2).setScaleB(1))
      conv1.add(ReLU(true))
      concat.add(conv1)
      val conv3 = Sequential()
      conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName("conv3x3_1").setScaleW(3).setScaleB(1.5))
      conv3.add(ReLU(true))
      conv3.add(SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setName("conv3x3_2").setScaleW(4).setScaleB(2))
      conv3.add(ReLU(true))
      concat.add(conv3)
      val conv5 = Sequential()
      conv5.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName("conv5x5_1").setScaleW(5).setScaleB(2.5))
      conv5.add(ReLU(true))
      conv5.add(SpatialConvolution(config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setName("conv5x5_2").setScaleW(6).setScaleB(3))
      conv5.add(ReLU(true))
      concat.add(conv5)
      val pool = Sequential()
      pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
        .setName("pool"))
      pool.add(SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1).setName("pool_conv")
        .setScaleW(7).setScaleB(3.5))
      pool.add(ReLU(true))
      concat.add(pool)
      concat
    }

    val model = inceptionLayerV1Seq(
      2, T(T(4), T(96, 128), T(16, 32), T(32)))
    model.freeze()
    Utils.getNamedModules(model).foreach(x => {
      if (!x._2.isInstanceOf[Container[_, _, _]]) {
        x._2.getScaleB() should be (0)
        x._2.getScaleW() should be (0)
      }
    })
    model.unFreeze()
    model("conv1x1").get.getScaleW() should be(2)
    model("conv1x1").get.getScaleB() should be(1)
    model("conv3x3_1").get.getScaleW() should be(3)
    model("conv3x3_1").get.getScaleB() should be(1.5)
    model("conv3x3_2").get.getScaleW() should be(4)
    model("conv3x3_2").get.getScaleB() should be(2)
    model("conv5x5_1").get.getScaleW() should be(5)
    model("conv5x5_1").get.getScaleB() should be(2.5)
    model("conv5x5_2").get.getScaleW() should be(6)
    model("conv5x5_2").get.getScaleB() should be(3)
    model("pool_conv").get.getScaleW() should be(7)
    model("pool_conv").get.getScaleB() should be(3.5)

    model.freeze("conv1x1", "conv3x3_1")
    model("conv1x1").get.getScaleW() should be(0)
    model("conv1x1").get.getScaleB() should be(0)
    model("conv3x3_1").get.getScaleW() should be(0)
    model("conv3x3_1").get.getScaleB() should be(0)

    model.unFreeze()
    model("conv1x1").get.getScaleW() should be(2)
    model("conv1x1").get.getScaleB() should be(1)
    model("conv3x3_1").get.getScaleW() should be(3)
    model("conv3x3_1").get.getScaleB() should be(1.5)
  }

  "get/set extra parameter" should "work fine" in {
    val bn = SpatialBatchNormalization(5)
    val model = Sequential()
        .add(SpatialConvolution(3, 5, 3, 3))
      .add(bn)
      .add(SpatialConvolution(5, 2, 3, 3))
      .add(BatchNormalization(2))

    val model2 = model.cloneModule()
    bn.runningMean.range(1, 5)
    model2 should not be (model)
    val extp = model.getExtraParameter()
    extp(0) should be (Tensor().range(1, 5))
    model2.setExtraParameter(extp)
    model2 should be (model)
  }

  "get/set extra parameter" should "work fine 2" in {
    val model = Sequential()
      .add(SpatialConvolution(3, 5, 3, 3))
      .add(SpatialConvolution(5, 2, 3, 3))

    val model2 = model.cloneModule()
    model2 should be (model)
    val extp = model.getExtraParameter()
    model2.setExtraParameter(extp)
    model2 should be (model)
  }

  "get/set extra parameter" should "work fine 3" in {
    val model = BatchNormalization(5)

    val model2 = model.cloneModule()
    model.runningMean.range(1, 5)
    model2 should not be (model)
    val extp = model.getExtraParameter()
    model2.setExtraParameter(extp)
    model2 should be (model)
  }

  "get/set extra parameter" should "work fine 4" in {
    val model = SpatialConvolution(3, 5, 3, 3)

    val model2 = model.cloneModule()
    model2 should be (model)
    val extp = model.getExtraParameter()
    model2.setExtraParameter(extp)
    model2 should be (model)
  }

  "Shallow copy" should "work properly" in {

    val linear = Linear[Float](2, 2)

    val shallowCopy = linear.clone(false).asInstanceOf[Linear[Float]]

    val originWeight = linear.weight

    val originBias = linear.bias

    originWeight.fill(1.0f)
    originBias.fill(2.0f)

    val input = Tensor[Float](2, 2).rand()

    val res1 = linear.forward(input)

    val res2 = shallowCopy.forward(input)

    res1 should be (res2)

  }

  "Deep copy" should  "work properly" in {

    val linear = Linear[Float](2, 2)

    val deepCopy = linear.clone(true).asInstanceOf[Linear[Float]]

    val input = Tensor[Float](2, 2).rand()

    val res1 = linear.forward(input)

    val res2 = deepCopy.forward(input)

    res1 should be(res2)
  }

  "Shallow copy for quantized model" should "work properly" in {
    val outputSize = 2
    val inputSize = 2

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f, 0.1f)

    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val weight = Tensor[Float](Storage(kernelData), 1, Array(outputSize, inputSize)).rand()
    val bias = Tensor[Float](Storage(biasData), 1, Array(outputSize)).rand()
    val linear = quantized.Linear[Float](outputSize, inputSize, initWeight = weight,
      initBias = bias).setName("quantLinear")

    val shallow = linear.clone(false).asInstanceOf[quantized.Linear[Float]]

    val res1 = linear.forward(input)

    val res2 = shallow.forward(input)

    res1 should be(res2)
  }

  "Deep copy for quantized model" should "work properly" in {
    val outputSize = 2
    val inputSize = 2

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f, 0.1f)

    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())

    val input2 = input.clone()

    val weight = Tensor[Float](Storage(kernelData), 1, Array(outputSize, inputSize)).rand()
    val bias = Tensor[Float](Storage(biasData), 1, Array(outputSize)).rand()
    val linear = quantized.Linear[Float](outputSize, inputSize, initWeight = weight,
      initBias = bias).setName("quantLinear")

    val deep = linear.clone(true).asInstanceOf[quantized.Linear[Float]]

    val res1 = linear.forward(input)

    val res2 = deep.forward(input2)

    res1 should be(res2)
  }
}
