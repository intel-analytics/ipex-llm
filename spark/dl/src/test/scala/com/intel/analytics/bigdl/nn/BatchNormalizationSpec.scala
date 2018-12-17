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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class BatchNormalizationSpec extends FlatSpec with Matchers with BeforeAndAfter{
  before {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("spark.master", "local[2]")
    Engine.init
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("spark.master")
  }

  "BacthNormalization parameter sync" should "work properly" in {
    val bn = BatchNormalization[Float](2)

    bn.setParallism(1)

    bn.weight.fill(1.0f)
    bn.bias.fill(1.0f)

    val input = Tensor[Float](2, 2)

    input.select(1, 1).fill(1.0f)
    input.select(1, 2).fill(2.0f)

    val gradOutput = Tensor[Float](2, 2)

    gradOutput.select(1, 1).fill(2.0f)
    gradOutput.select(1, 2).fill(1.0f)

    val output = bn.forward(input)

    val gradInput = bn.backward(input, gradOutput)

    val saveMean = bn.saveMean
    val saveStd = bn.saveStd
    val runningMean = bn.runningMean
    val runningVar = bn.runningVar

    val bn1 = BatchNormalization[Float](2)

    bn1.setParallism(2)

    bn1.weight.fill(1.0f)
    bn1.bias.fill(1.0f)

    val bn2 = bn1.cloneModule().asInstanceOf[BatchNormalization[Float]]

    val modules = Array(bn1, bn2)

    val input1 = Tensor[Float](1, 2).fill(1.0f)

    val input2 = Tensor[Float](1, 2).fill(2.0f)

    val inputs = Array(input1, input2)

    val gradOutput1 = Tensor[Float](1, 2).fill(2.0f)
    val gradOutput2 = Tensor[Float](1, 2).fill(1.0f)

    val gradOutputs = Array(gradOutput1, gradOutput2)

    Engine.default.invokeAndWait2((0 until modules.size).map(i =>
      () => {
        val trainStart = System.nanoTime()
        val sub = modules(i)
        val subInput = inputs(i)
        val subGradOutput = gradOutputs(i)
        sub.forward(subInput)
        sub.backward(subInput, subGradOutput)
      }
    ))

    val saveMean1 = bn1.saveMean
    val saveStd1 = bn1.saveStd
    val runningMean1 = bn1.runningMean
    val runningVar1 = bn1.runningVar
    val gradInput1 = bn1.gradInput
    val out1 = bn1.output.squeeze

    val saveMean2 = bn2.saveMean
    val saveStd2 = bn2.saveStd
    val runningMean2 = bn2.runningMean
    val runningVar2 = bn2.runningVar
    val gradInput2 = bn2.gradInput
    val out2 = bn2.output.squeeze()

    saveMean should be (saveMean1)
    saveMean should be (saveMean2)
    saveStd should be (saveStd1)
    saveStd should be (saveStd2)
    runningMean should be (runningMean1)
    runningMean should be (runningMean2)
    runningVar should be (runningVar1)
    runningVar should be (runningVar2)

    val sout1 = output.select(1, 1).squeeze()
    sout1  should be (out1)

    val sout2 = output.select(1, 2).squeeze()
    sout2 should be (bn2.output)

    val gin1 = gradInput.select(1, 1)

    val gin2 = gradInput.select(1, 2)

    gin1.squeeze should be (gradInput1.squeeze)
    gin2.squeeze should be (gradInput2.squeeze)
  }

  "A BatchNormalization" should "generate correct output using default arguments" in {
    val bn = BatchNormalization[Double](None)
    val input = Tensor[Double](3, 3)

    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    val mean = Tensor[Double](Storage[Double](Array(4.0, 5.0, 6.0)))
    val std = Tensor(Storage(Array(0.4082479, 0.4082479, 0.4082479)))
    val output1 = Tensor[Double](3, 3)
    for (i <- 1 to 3) {
      for (j <- 1 to 3) {
        output1.setValue(i, j, (input(Array(i, j)) - mean(Array(j))) * std(Array(j)))
      }
    }

    output.nDimension() should be(2)
    output.size(1) should be(3)
    output.size(2) should be(3)

    output.map(output1, (a, b) => {
      a should be (b +- 0.0001)
      a
    })
  }

  "A BatchNormalization" should "generate correct output" in {

    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)

    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    output.nDimension() should be(2)
    output.size(1) should be(3)
    output.size(2) should be(3)
    output(Array(1, 1)) should be(-0.0225 +- 0.0001)
    output(Array(1, 2)) should be(-0.0449 +- 0.0001)
    output(Array(1, 3)) should be(-0.0674 +- 0.0001)
    output(Array(2, 1)) should be(0.1 +- 0.0001)
    output(Array(2, 2)) should be(0.2 +- 0.0001)
    output(Array(2, 3)) should be(0.3 +- 0.0001)
    output(Array(3, 1)) should be(0.2225 +- 0.0001)
    output(Array(3, 2)) should be(0.4449 +- 0.0001)
    output(Array(3, 3)) should be(0.6674 +- 0.0001)
  }

  "A BatchNormalization" should "generate correct output for given weight and bias" in {
    val weight = Tensor[Double](T(0.1, 0.2, 0.3))
    val bias = Tensor[Double](T(0.1, 0.2, 0.3))
    val bn = new BatchNormalization[Double](nOutput = 3, initWeight = weight, initBias = bias)
    val input = Tensor[Double](3, 3)

    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    output.nDimension() should be(2)
    output.size(1) should be(3)
    output.size(2) should be(3)
    output(Array(1, 1)) should be(-0.0225 +- 0.0001)
    output(Array(1, 2)) should be(-0.0449 +- 0.0001)
    output(Array(1, 3)) should be(-0.0674 +- 0.0001)
    output(Array(2, 1)) should be(0.1 +- 0.0001)
    output(Array(2, 2)) should be(0.2 +- 0.0001)
    output(Array(2, 3)) should be(0.3 +- 0.0001)
    output(Array(3, 1)) should be(0.2225 +- 0.0001)
    output(Array(3, 2)) should be(0.4449 +- 0.0001)
    output(Array(3, 3)) should be(0.6674 +- 0.0001)
  }

  "A BatchNormalization" should "generate correct gradient" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    val gradOutput = Tensor[Double](3, 3)
    var j = 0.0
    gradOutput.apply1(e => {
      j += 0.1; j
    })
    val gradInput = bn.backward(input, gradOutput)

    gradInput.nDimension() should be(2)
    gradInput.size(1) should be(3)
    gradInput.size(2) should be(3)

    gradInput(Array(1, 1)) should be(-2.0412e-8 +- 1e-12)
    gradInput(Array(1, 2)) should be(-4.0825e-8 +- 1e-12)
    gradInput(Array(1, 3)) should be(-6.1237e-8 +- 1e-12)
    gradInput(Array(2, 1)) should be(-0.0 +- 0.0001)
    gradInput(Array(2, 2)) should be(-0.0 +- 0.0001)
    gradInput(Array(2, 3)) should be(-0.0 +- 0.0001)
    gradInput(Array(3, 1)) should be(2.0412e-8 +- 1e-12)
    gradInput(Array(3, 2)) should be(4.0825e-8 +- 1e-12)
    gradInput(Array(3, 3)) should be(6.1237e-8 +- 1e-12)

    bn.gradWeight.nDimension() should be(1)
    bn.gradWeight.size(1) should be(3)
    bn.gradWeight(Array(1)) should be(0.7348 +- 0.0001)
    bn.gradWeight(Array(2)) should be(0.7348 +- 0.0001)
    bn.gradWeight(Array(3)) should be(0.7348 +- 0.0001)

    bn.gradBias.nDimension() should be(1)
    bn.gradBias.size(1) should be(3)
    bn.gradBias(Array(1)) should be(1.2 +- 0.0001)
    bn.gradBias(Array(2)) should be(1.5 +- 0.0001)
    bn.gradBias(Array(3)) should be(1.8 +- 0.0001)
  }

  "A BatchNormalization evaluating" should "generate correct output" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    var output = bn.forward(input)

    val gradOutput = Tensor[Double](3, 3)
    var j = 0.0
    gradOutput.apply1(e => {
      j += 0.1; j
    })
    val gradInput = bn.backward(input, gradOutput)
    bn.evaluate()
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
  }

  it should "generate correct output for no batch" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    bn.evaluate()

    val input = Tensor[Double](3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)
    output.valueAt(1) should be(0.2 +- 0.00001)
    output.valueAt(2) should be(0.6 +- 0.00001)
    output.valueAt(3) should be(1.2 +- 0.00001)
  }

  "A BatchNormalization with scaleW and scaleB" should "generate correct output" in {
    val weight = Tensor[Double](T(0.1, 0.2, 0.3))
    val bias = Tensor[Double](T(0.1, 0.2, 0.3))
    val bn1 = new BatchNormalization[Double](nOutput = 3, initWeight = weight, initBias = bias)
    val bn2 = bn1.cloneModule().asInstanceOf[BatchNormalization[Double]].setScaleW(0.5).setScaleB(2)
    val input = Tensor[Double](3, 3)

    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output1 = bn1.forward(input)
    val output2 = bn2.forward(input)
    output1 should be(output2)

    val gradOutput = Tensor(output1)
    val gradInput1 = bn1.backward(input, gradOutput)
    val gradInput2 = bn2.backward(input, gradOutput)
    gradInput1 should be(gradInput2)

    bn2.gradWeight should be(bn1.gradWeight.mul(0.5))
    bn2.gradBias should be(bn1.gradBias.mul(2))
  }

  "BatchNormalization backward" should "be good when affine is false" in {
    val layer = BatchNormalization[Float](3, affine = false)
    val input = Tensor[Float](4, 3).fill(1)
    val gradOutput = Tensor[Float](4, 3).fill(1)
    val output = layer.forward(input)
    output should be(Tensor[Float](4, 3).fill(0))
    val gradInput = layer.backward(input, gradOutput)
    gradInput should be(Tensor[Float](4, 3).fill(0))
  }
}

class BatchNormalizationSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val batchNorm = BatchNormalization[Float](5).setName("batchNorm")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(batchNorm, input)
  }
}
