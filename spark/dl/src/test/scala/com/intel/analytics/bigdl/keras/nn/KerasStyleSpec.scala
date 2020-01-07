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

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.nn.abstractnn.InvalidLayer
import com.intel.analytics.bigdl.nn.keras.{Activation, Convolution1D, Dense, GlobalMaxPooling1D, Input, InputLayer, KerasIdentityWrapper, Model, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.mkldnn.Equivalent
import com.intel.analytics.bigdl.nn.{Input => TInput, Sequential => TSequential, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, RandomGenerator, Shape, T, TestUtils}


class KerasStyleSpec extends BigDLSpecHelper {

  "Graph: Dense" should "work correctly" in {
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20, activation = "relu").setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = model.forward(inputData)
    require(model.getOutputShape().toSingle().sameElements(Array(-1, 5)))
    require(model.getInputShape().toSingle().sameElements(Array(-1, 10)))
  }

  "Sequential: Dense" should "work correctly" in {
    val seq = KSequential[Float]()
    val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
    val d2 = Dense[Float](5).setName("dense2")
    val d3 = Dense[Float](6).setName("dense4")
    seq.add(d1)
    seq.add(d2)
    seq.add(d3)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(d3.getOutputShape().toSingle().sameElements(Array(-1, 6)))
    require(d3.getInputShape().toSingle().sameElements(Array(-1, 5)))
  }

  "Frozen sequential" should "be tested" in {
    intercept[RuntimeException] {
      val seq = KSequential[Float]()
      val seq1 = KSequential[Float]()
      seq.add(seq1)
      seq1.add(Dense[Float](20, inputShape = Shape(10)))
    }
  }

  "Sequential: shared relu" should "not work correctly" in {
    val thrown = intercept[Exception] {
      val sharedRelu = new KerasIdentityWrapper(ReLU[Float]())
      val seq1 = KSequential[Float]()
      seq1.add(Dense[Float](20, inputShape = Shape(10)))
      seq1.add(sharedRelu)
      assert(seq1.getOutputShape().toSingle().sameElements(Array(-1, 20)))

      val seq2 = KSequential[Float]()
      seq2.add(Dense[Float](5, inputShape = Shape(20)))
      seq2.add(sharedRelu)
      assert(seq2.getOutputShape().toSingle().sameElements(Array(-1, 5)))

      val seq = KSequential[Float]()
      seq.add(seq1)
      seq.add(seq2)

      val inputData = Tensor[Float](Array(20, 10)).rand()
      val output = seq.forward(inputData)
      assert(seq.getInputShape().toSingle().sameElements(Array(-1, 10)))
      assert(seq.getOutputShape().toSingle().sameElements(Array(-1, 5)))
    }
    assert(thrown.getMessage().contains("multiple times"))
  }

  "Graph: shared relu" should "not work correctly" in {
    val thrown = intercept[Exception] {
      val input = Input(inputShape = Shape(10, 20))
      val sharedRelu = new Activation("relu")
      val out1 = sharedRelu.inputs(input)

      val seq = KSequential[Float]()
      seq.add(InputLayer(inputShape = Shape(10, 20)))
      seq.add(sharedRelu)
      val out2 = seq.inputs(out1)
      val model = Model(input, out2)
    }
    assert(thrown.getMessage().contains("multiple times"))
  }

  "Graph: shared relu as dest" should "not work correctly" in {
    val thrown = intercept[Exception] {
      val input = Input(inputShape = Shape(10, 20))
      val sharedRelu = new Activation("relu")
      val out1 = sharedRelu.inputs(input)
      val out2 = sharedRelu.inputs(Input(inputShape = Shape(10, 20)))
    }
    assert(thrown.getMessage().contains("multiple times"))
  }

  "TSequential" should "work with alex" in {
    val model = AlexNet_OWT(1000, false, true)
    TSequential[Float].add(model)
  }

  "TSequential" should "not work with dense" in {
    intercept[InvalidLayer] {
      val seq = TSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq.add(d1)
    }
  }

  "Incompatible inputShape" should "not work" in {
    intercept[RuntimeException] {
      val seq = KSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq.add(InputLayer(inputShape = Shape(5)))
      seq.add(d1)
    }
  }

  "TGraph" should "not work with dense" in {
    intercept[InvalidLayer] {
      val d1 = Dense[Float](20).setName("dense1").inputs(Input(inputShape = Shape(10)))
      val l1 = Linear(2, 3).inputs(d1)
    }
  }

  "KGraph" should "not work with shared layers" in {
    val thrown = intercept[RuntimeException] {
      val input = Input(inputShape = Shape(10))
      val dense1 = Dense(10, inputShape = Shape(10))
      val node1 = dense1.inputs(input)
      val seq = KSequential[Float]().add(dense1).inputs(node1)
      Model(input, seq)
    }
    assert(thrown.getMessage().contains("multiple times"))
  }

  "KGraph" should "not work with shared weights" in {
    val thrown = intercept[RuntimeException] {
      val input1 = Input(inputShape = Shape(10))
      val input2 = Input(inputShape = Shape(10))
      val l = Dense(10, inputShape = Shape(10))
      val node1 = l.inputs(input1)
      val node2 = l.inputs(input2)
      Model(Array(input1, input2), Array(node1, node2))
    }
    assert(thrown.getMessage().contains("multiple times"))
  }

  "KGraph" should "work with shared input" in {
    val input1 = Input(inputShape = Shape(10))
    val l1 = Dense(10, inputShape = Shape(10))
    val l2 = Dense(10, inputShape = Shape(10))
    val node1 = l1.inputs(input1)
    val node2 = l2.inputs(input1)
    Model(input1, Array(node1, node2))
  }

  "Torch style linear and seq and linear" should "not work with keras Model" in {
    intercept[InvalidLayer] {
      val input = Input(inputShape = Shape(10))
      val l1 = Linear(10, 3).inputs(input)
      val seq = TSequential[Float]().inputs(l1)
      val l2 = Linear(3, 4).inputs(seq)
      Model(input, l2)
    }
  }

  "Torch style inputs in Model constructor" should "not work" in {
    intercept[InvalidLayer] {
      val tinput = TInput()
      val l1 = Linear(10, 3).inputs(tinput)
      Model(tinput, l1)
    }
  }

  "TSequential" should "not works with container containing Dense" in {
    val seq = TSequential[Float]()
    intercept[InvalidLayer] {
      val parallelTable = ParallelTable[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      parallelTable.add(d1)
      seq.add(parallelTable)
    }
  }

  "TSequential" should "not work with container with dense" in {
    intercept[InvalidLayer] {
      val seq = TSequential[Float]()
      val seq2 = TSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq2.add(d1)
      seq.add(seq2)
    }
  }

  "save and reload model" should "work correctly" in {
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20).setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    tmpFile.delete()
    model.saveModule(absPath)
    val reloadedModel = Module.loadModule(absPath)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = reloadedModel.forward(inputData)
  }

  "save and reload sequential" should "work correctly" in {
    val kseq = KSequential[Float]()
    val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
    val d2 = Dense[Float](5).setName("dense2")
    kseq.add(d1)
    kseq.add(d2)
    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    tmpFile.delete()
    kseq.saveModule(absPath)
    val reloadedModel = Module.loadModule(absPath)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = reloadedModel.forward(inputData)
  }

  "multiple outputs with index" should "be test" in {
    val input1 = Input[Float](inputShape = Shape(10))
    val input2 = Input[Float](inputShape = Shape(10))
    val d1 = Dense[Float](20).setName("dense1").inputs(input1)
    val d2 = Dense[Float](5).setName("dense2").inputs(input2)
    val multiOutputGraph = Model[Float](Array(input1, input2), Array(d1, d2))

    val input3 = Input[Float](inputShape = Shape(10))
    val input4 = Input[Float](inputShape = Shape(10))
    val multiOutput = multiOutputGraph.inputs(Array(input3, input4))

    val relu1 = Activation[Float]("relu").inputs(multiOutput(1))
    val model = Model[Float](Array(input3, input4), relu1)
    model.forward(T(Tensor[Float](Array(2, 10)).rand(), Tensor[Float](Array(2, 10)).rand()))
    assert(model.getOutputShape().toSingle().sameElements(Array(-1, 20)))
  }

  "Empty inputs is not allow" should "be test" in {
    val thrown = intercept[Exception] {
      val d1 = Dense[Float](20).setName("dense1").inputs()
    }
    assert(thrown.getMessage().contains("Empty input is not allow"))
  }

  "InputLayer" should "be test" in {
    val inputLayer = InputLayer(inputShape = Shape(2, 3))
    val seq = KSequential[Float]()
    seq.add(inputLayer)
    val inputData = Tensor[Float](Array(2, 2, 3)).rand()
    val output = seq.forward(inputData)
    seq.forward(output)
    TestUtils.compareOutputShape(seq, Shape(2, 3)) should be (true)
  }

  "KSequential to IRGraph" should "work" in {
    System.setProperty("bigdl.engineType", "mkldnn")

    RandomGenerator.RNG.setSeed(10)
    import com.intel.analytics.bigdl.mkl.Memory

    val seq = KSequential[Float]()
    seq.add(InputLayer(inputShape = Shape(20, 100)))
    seq.add(Convolution1D(10, 5, activation = "relu"))
    seq.add(GlobalMaxPooling1D())
    seq.add(Dense(128))
    // seq.add(KDropout(0.2))
    seq.add(Activation("relu"))
    seq.add(Dense(10, activation = "softmax"))

    val graph = seq.toGraph().asInstanceOf[StaticGraph[Float]]
    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.ntc))
    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val ir = graph.asInstanceOf[StaticGraph[Float]].cloneModule().toIRgraph()

    val tensor = Tensor[Float](Array(3, 20, 100)).rand()
    val outputBlas = graph.forward(tensor)
    val output = ir.forward(tensor)
    outputBlas should be(output)

    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
    val gradInputBlas = graph.backward(tensor, gradOutput)
    val gradInput = ir.backward(tensor, gradOutput)

    Equivalent.nearequals(gradInput.toTensor[Float],
      gradInputBlas.toTensor[Float], 1e-5) should be(true)

    System.clearProperty("bigdl.engineType")
  }

  "KGraph to IRGraph" should "work" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    import com.intel.analytics.bigdl.mkl.Memory
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20, activation = "relu").setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)

    val graph = model.toGraph().asInstanceOf[StaticGraph[Float]]
    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val ir = graph.asInstanceOf[StaticGraph[Float]].cloneModule().toIRgraph()

    val tensor = Tensor[Float](Array(3, 10)).rand()
    val outputBlas = graph.forward(tensor)
    val output = ir.forward(tensor)

    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
    val gradInputBlas = graph.backward(tensor, gradOutput)
    val gradInput = ir.backward(tensor, gradOutput)

    outputBlas should be(output)
    gradInputBlas should be(gradInput)

    System.clearProperty("bigdl.engineType")
  }

  "KSequential with KGraph module to IRGraph" should "work" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    import com.intel.analytics.bigdl.mkl.Memory
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20, activation = "relu").setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val kgraph = Model[Float](input, d2)

    val seq = KSequential[Float]()
    seq.add(kgraph)

    val graph = seq.toGraph().asInstanceOf[StaticGraph[Float]]
    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val ir = graph.asInstanceOf[StaticGraph[Float]].toIRgraph()

    val tensor = Tensor[Float](Array(3, 10)).rand()

    val outputBlas = graph.forward(tensor)
    val output = ir.forward(tensor)

    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
    val gradInputBlas = graph.backward(tensor, gradOutput)
    val gradInput = ir.backward(tensor, gradOutput)

    outputBlas should be(output)
    gradInputBlas should be(gradInput)

    System.clearProperty("bigdl.engineType")
  }

  "KGraph with KGraph module to IRGraph" should "work" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    import com.intel.analytics.bigdl.mkl.Memory
    val input1 = Input[Float](inputShape = Shape(10))

    val input2 = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20, activation = "relu").setName("dense1").inputs(input2)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val kgraph1 = Model[Float](input2, d2).inputs(input1)

    val kgraph2 = Model[Float](input1, kgraph1)

    val graph = kgraph2.toGraph().asInstanceOf[StaticGraph[Float]]
    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val ir = graph.asInstanceOf[StaticGraph[Float]].toIRgraph()

    val tensor = Tensor[Float](Array(3, 10)).rand()
    val outputBlas = graph.forward(tensor)
    val output = ir.forward(tensor)

    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
    val gradInputBlas = graph.backward(tensor, gradOutput)
    val gradInput = ir.backward(tensor, gradOutput)

    outputBlas should be(output)
    gradInputBlas should be(gradInput)

    System.clearProperty("bigdl.engineType")
  }
}
