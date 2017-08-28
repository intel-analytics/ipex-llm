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
package com.intel.analytics.bigdl.utils.serializer

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.{VolumetricFullConvolution, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag
import scala.util.Random

class ModuleSerializerSpec extends FlatSpec with Matchers {
  "Abs serializer" should "work properly" in {
    val abs = Abs().setName("abs")
    val tensor1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = abs.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/abs.bigdl", abs, true)
    val loadedModule = ModuleLoader.loadFromFile("/tmp/abs.bigdl")
    val res2 = loadedModule.forward(tensor2)
    res1 should be (res2)
  }

  "Add serializer" should "work properly" in {
    val add = Add(5)
    val tensor1 = Tensor(5).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = add.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/add.bigdl", add, true)
    val loadedAdd = ModuleLoader.loadFromFile("/tmp/add.bigdl")
    val res2 = loadedAdd.forward(tensor2)
    res1 should be (res2)
  }

  "AddConst serializer" should "work properly" in {
    val addconst = AddConstant(5)
    val tensor1 = Tensor(5).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = addconst.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/addconst.bigdl", addconst, true)
    val loadedAddConst = ModuleLoader.loadFromFile("/tmp/addconst.bigdl")
    val res2 = loadedAddConst.forward(tensor2)
    res1 should be (res2)
  }

  "BatchNormalization serializer" should "work properly" in {
    val batchNorm = BatchNormalization(5)
    val tensor1 = Tensor(2, 5).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = batchNorm.forward(tensor1)
    ModulePersister.saveToFile("/tmp/batchNorm.bigdl", batchNorm, true)
    val loadedBatchNorm = ModuleLoader.loadFromFile("/tmp/batchNorm.bigdl")
    val res2 = loadedBatchNorm.forward(tensor2)
    res1 should be (res2)
  }

  "BiLinear serializer" should "work properly" in {
    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 3).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val biLinear = Bilinear(5, 3, 2)
    val res1 = biLinear.forward(input)
    ModulePersister.saveToFile("/tmp/biLinear.bigdl", biLinear, true)
    val loadedBiLinear = ModuleLoader.loadFromFile("/tmp/biLinear.bigdl")
    val res2 = loadedBiLinear.forward(input)
    res1 should be (res2)
  }

  "BinaryTreeLSTM serializer" should " work properly" in {

    RNG.setSeed(1000)
    val binaryTreeLSTM = BinaryTreeLSTM(2, 2)

    val inputs =
      Tensor(
        T(T(T(1f, 2f),
          T(2f, 3f),
          T(4f, 5f))))

    val tree =
      Tensor(
        T(T(T(2f, 5f, -1f),
          T(0f, 0f, 1f),
          T(0f, 0f, 2f),
          T(0f, 0f, 3f),
          T(3f, 4f, 0f))))

    val input = T(inputs, tree)

    val res1 = binaryTreeLSTM.forward(input)
    val res11 = binaryTreeLSTM.forward(input)
    res1 should be (res11)
    ModulePersister.saveToFile("/tmp/binaryTreeLSTM.bigdl", binaryTreeLSTM, true)
    RNG.setSeed(1000)
    val loadedBinaryTreeLSTM = ModuleLoader.loadFromFile("/tmp/binaryTreeLSTM.bigdl")
    val res2 = loadedBinaryTreeLSTM.forward(input)
    res1 should be (res2)

  }

  "BiRecurrent serializer" should "work properly" in {
    val input1 = Tensor(1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    RNG.setSeed(100)
    val biRecurrent = BiRecurrent().add(RnnCell(6, 4, Sigmoid()))
    val res1 = biRecurrent.forward(input1)
    ModulePersister.saveToFile("/tmp/biRecurrent.bigdl", biRecurrent, true)
    RNG.setSeed(100)
    val loadedRecurent = ModuleLoader.loadFromFile("/tmp/biRecurrent.bigdl")
    val res2 = loadedRecurent.forward(input2)
    res1 should be (res2)
  }

  "BiRecurrent serializer" should "work properly with BatchNormParams" in {
    val input1 = Tensor(1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    RNG.setSeed(100)
    val biRecurrent = BiRecurrent(batchNormParams = BatchNormParams()).add(RnnCell(6, 4, Sigmoid()))
    val res1 = biRecurrent.forward(input1)
    ModulePersister.saveToFile("/tmp/biRecurrent.bigdl", biRecurrent, true)
    RNG.setSeed(100)
    val loadedRecurent = ModuleLoader.loadFromFile("/tmp/biRecurrent.bigdl")
    val res2 = loadedRecurent.forward(input2)
    res1 should be (res2)
  }

  "BiRecurrent serializer" should "work properly with isSplitInput" in {
    val input1 = Tensor(1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    RNG.setSeed(100)
    val biRecurrent = BiRecurrent(isSplitInput = false).add(RnnCell(6, 4, Sigmoid()))
    val res1 = biRecurrent.forward(input1)
    ModulePersister.saveToFile("/tmp/biRecurrent.bigdl", biRecurrent, true)
    RNG.setSeed(100)
    val loadedRecurent = ModuleLoader.loadFromFile("/tmp/biRecurrent.bigdl")
    val res2 = loadedRecurent.forward(input2)
    res1 should be (res2)
  }

  "Bottle serializer" should "work properly" in {
    val input1 = Tensor(10).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)

    val bottle = new Bottle(Linear(10, 2).asInstanceOf[Module[Float]], 2, 2)

    val res1 = bottle.forward(input1)
    ModulePersister.saveToFile("/tmp/bottle.bigdl", bottle, true)
    val loadedBottle = ModuleLoader.loadFromFile("/tmp/bottle.bigdl")
    val res2 = loadedBottle.forward(input2)
    res1 should be (res2)
  }

  "Caddserializer" should "work properly" in {
    val input1 = Tensor(5, 1).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val cadd = CAdd(Array(5, 1))
    val res1 = cadd.forward(input1)
    ModulePersister.saveToFile("/tmp/cadd.bigdl", cadd, true)
    val loadedCadd = ModuleLoader.loadFromFile("/tmp/cadd.bigdl")
    val res2 = loadedCadd.forward(input2)
    res1 should be (res2)
  }

  "CaddTable serializer" should "work properly" in {
    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val caddTable = CAddTable(false)

    val res1 = caddTable.forward(input)
    ModulePersister.saveToFile("/tmp/caddTable.bigdl", caddTable, true)
    val loadedCaddTable = ModuleLoader.loadFromFile("/tmp/caddTable.bigdl")
    val res2 = loadedCaddTable.forward(input)
    res1 should be (res2)
  }


  "CDivTable serializer" should "work properly" in {
    val cdivTable = new CDivTable()
    val input1 = Tensor(10).apply1(e => Random.nextFloat())
    val input2 = Tensor(10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = cdivTable.forward(input)

    ModulePersister.saveToFile("/tmp/cdivTable.bigdl", cdivTable, true)
    val loadedCdivTable = ModuleLoader.loadFromFile("/tmp/cdivTable.bigdl")
    val res2 = cdivTable.forward(input)
    res1 should be (res2)
  }

  "Clamp serializer" should "work properly" in {

    val input1 = Tensor(10).apply1(e => Random.nextFloat())

    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)

    val clamp = Clamp(1, 10)
    val res1 = clamp.forward(input1)

    ModulePersister.saveToFile("/tmp/clamp.bigdl", clamp, true)
    val loadedClamp = ModuleLoader.loadFromFile("/tmp/clamp.bigdl")
    val res2 = loadedClamp.forward(input2)
    res1 should be (res2)
  }

  "CMaxTable serializer" should "work properly" in {
    val cmaxTable = new CMaxTable()
    val input1 = Tensor(10).apply1(e => Random.nextFloat())
    val input2 = Tensor(10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = cmaxTable.forward(input)

    ModulePersister.saveToFile("/tmp/cmaxTable.bigdl", cmaxTable, true)
    val loadedCmaxTable = ModuleLoader.loadFromFile("/tmp/cmaxTable.bigdl")
    val res2 = loadedCmaxTable.forward(input)
    res1 should be (res2)
  }

  "CMinTable serializer" should "work properly" in {
    val cminTable = new CMinTable()
    val input1 = Tensor(10).apply1(e => Random.nextFloat())
    val input2 = Tensor(10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = cminTable.forward(input)

    ModulePersister.saveToFile("/tmp/cminTable.bigdl", cminTable, true)
    val loadedCminTable = ModuleLoader.loadFromFile("/tmp/cminTable.bigdl")
    val res2 = loadedCminTable.forward(input)
    res1 should be (res2)
  }

  "CMulserializer" should "work properly" in {
    val input1 = Tensor(5, 1).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)

    val cmul = CMul(Array(5, 1))

    val res1 = cmul.forward(input1)
    ModulePersister.saveToFile("/tmp/cmul.bigdl", cmul, true)
    val loadedCmul = ModuleLoader.loadFromFile("/tmp/cmul.bigdl")
    val res2 = loadedCmul.forward(input2)
    res1 should be (res2)
  }

  "CMulTable serializer" should "work properly" in {
    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val cmulTable = CMulTable()

    val res1 = cmulTable.forward(input)
    ModulePersister.saveToFile("/tmp/cmulTable.bigdl", cmulTable, true)
    val loadedCmulTable = ModuleLoader.loadFromFile("/tmp/cmulTable.bigdl")
    val res2 = loadedCmulTable.forward(input)
    res1 should be (res2)
  }

  "Concatserializer" should "work properly" in {
    val input1 = Tensor(2, 2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)

    val concat = Concat(2)

    concat.add(Abs())
    concat.add(Abs())

    val res1 = concat.forward(input1)
    ModulePersister.saveToFile("/tmp/concat.bigdl", concat, true)
    val loadedConcat = ModuleLoader.loadFromFile("/tmp/concat.bigdl")
    val res2 = loadedConcat.forward(input2)
    res1 should be (res2)
  }

  "ConcatTable serializer" should "work properly" in {
    val concatTable = ConcatTable()
    concatTable.add(Linear(10, 2))
    concatTable.add(Linear(10, 2))

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = concatTable.forward(tensor1)

    ModulePersister.saveToFile("/tmp/concatTable.bigdl", concatTable, true)
    val loadedConcatTable = ModuleLoader.loadFromFile("/tmp/concatTable.bigdl")
    val res2 = loadedConcatTable.forward(tensor2)
    res1 should be (res2)
  }

  "Contiguous serializer" should "work properly" in {
    val contiguous = Contiguous()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = contiguous.forward(tensor1)

    ModulePersister.saveToFile("/tmp/contiguous.bigdl", contiguous, true)
    val loadedContiguous = ModuleLoader.loadFromFile("/tmp/contiguous.bigdl")
    val res2 = loadedContiguous.forward(tensor2)
    res1 should be (res2)
  }

  "ConvLSTMPeephole2D serializer" should " work properly" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val convLSTMPeephole2d = Recurrent()
    val model = Sequential()
      .add(convLSTMPeephole2d
        .add(ConvLSTMPeephole(
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1,
          withPeephole = false)))
      .add(View(hiddenSize * kernalH * kernalW))

    val input1 = Tensor(batchSize, seqLength, inputSize, kernalW, kernalH).rand

    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = convLSTMPeephole2d.forward(input1)
    ModulePersister.saveToFile("/tmp/convLSTMPeephole2d.bigdl", convLSTMPeephole2d, true)
    val loadedConvLSTMPeephole2d = ModuleLoader.loadFromFile("/tmp/convLSTMPeephole2d.bigdl")
    val res2 = loadedConvLSTMPeephole2d.forward(input2)
    res1 should be (res2)
  }

  "ConvLSTMPeephole3D serializer" should " work properly" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val convLSTMPeephole3d = Recurrent()
    val model = Sequential()
      .add(convLSTMPeephole3d
        .add(ConvLSTMPeephole3D(
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1,
          withPeephole = false)))
      .add(View(hiddenSize * kernalH * kernalW))

    val input1 = Tensor(batchSize, seqLength, inputSize, kernalW, kernalH, 3).rand

    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = convLSTMPeephole3d.forward(input1)
    ModulePersister.saveToFile("/tmp/convLSTMPeephole3d.bigdl", convLSTMPeephole3d, true)
    val loadedConvLSTMPeephole3d = ModuleLoader.loadFromFile("/tmp/convLSTMPeephole3d.bigdl")
    val res2 = loadedConvLSTMPeephole3d.forward(input2)
    res1 should be (res2)
  }

  "Cosine serializer" should "work properly" in {
    val cosine = Cosine(5, 5)

    val tensor1 = Tensor(5).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = cosine.forward(tensor1)

    ModulePersister.saveToFile("/tmp/cosine.bigdl", cosine, true)
    val loadedCosine = ModuleLoader.loadFromFile("/tmp/cosine.bigdl")
    val res2 = loadedCosine.forward(tensor2)
    res1 should be (res2)
  }

  "CosineDistance serializer" should "work properly" in {
    val cosineDistance = CosineDistance()

    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = cosineDistance.forward(input)

    ModulePersister.saveToFile("/tmp/cosineDistance.bigdl", cosineDistance, true)
    val loadedCosineDistance = ModuleLoader.loadFromFile("/tmp/cosineDistance.bigdl")
    val res2 = loadedCosineDistance.forward(input)
    res1 should be (res2)
  }

  "CSubTable serializer" should "work properly" in {
    val csubTable = CSubTable()

    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = csubTable.forward(input)

    ModulePersister.saveToFile("/tmp/csubTable.bigdl", csubTable, true)
    val loadedCSubTable = ModuleLoader.loadFromFile("/tmp/csubTable.bigdl")
    val res2 = loadedCSubTable.forward(input)
    res1 should be (res2)
  }

  "Dotproduct serializer" should "work properly" in {

    val dotProduct = DotProduct()

    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = dotProduct.forward(input)

    ModulePersister.saveToFile("/tmp/dotProduct.bigdl", dotProduct, true)
    val loadedDotProduct = ModuleLoader.loadFromFile("/tmp/dotProduct.bigdl")
    val res2 = loadedDotProduct.forward(input)
    res1 should be (res2)
  }

  "Dropout serializer" should "work properly" in {
    RNG.setSeed(100)
    val dropout = Dropout()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = dropout.forward(tensor1)

    ModulePersister.saveToFile("/tmp/dropout.bigdl", dropout, true)
    RNG.setSeed(100)
    val loadedDropout = ModuleLoader.loadFromFile("/tmp/dropout.bigdl")
    val res2 = loadedDropout.forward(tensor2)
    res1 should be (res2)
  }

  "Echo serializer " should " work properly" in {
    val echo = Echo()
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = echo.forward(tensor1)
    ModulePersister.saveToFile("/tmp/echo.bigdl", echo, true)
    val loadedEcho = ModuleLoader.loadFromFile("/tmp/echo.bigdl")
    val res2 = loadedEcho.forward(tensor2)
    res1 should be (res2)
  }

  "ELU serializer" should "work properly" in {
    val elu = ELU()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = elu.forward(tensor1)

    ModulePersister.saveToFile("/tmp/elu.bigdl", elu, true)
    val loadedElu = ModuleLoader.loadFromFile("/tmp/elu.bigdl")
    val res2 = loadedElu.forward(tensor2)
    res1 should be (res2)
  }

  "Euclidena serializer" should "work properly" in {
    val euclidean = Euclidean(7, 7)

    val tensor1 = Tensor(8, 7).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = euclidean.forward(tensor1)

    ModulePersister.saveToFile("/tmp/euclidean.bigdl", euclidean, true)
    val loadedEuclidean = ModuleLoader.loadFromFile("/tmp/euclidean.bigdl")
    val res2 = loadedEuclidean.forward(tensor2)
    res1 should be (res2)
  }

  "Exp serializer" should "work properly" in {
    val exp = Exp()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = exp.forward(tensor1)

    ModulePersister.saveToFile("/tmp/exp.bigdl", exp, true)
    val loadedExp = ModuleLoader.loadFromFile("/tmp/exp.bigdl")
    val res2 = loadedExp.forward(tensor2)
    res1 should be (res2)
  }

  "FlattenTable serializer" should "work properly" in {
    val flattenTable = FlattenTable()

    val input1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = flattenTable.forward(input)

    ModulePersister.saveToFile("/tmp/flattenTable.bigdl", flattenTable, true)
    val loadedFlattenTable = ModuleLoader.loadFromFile("/tmp/flattenTable.bigdl")
    val res2 = loadedFlattenTable.forward(input)
    res1 should be (res2)
  }

  "GradientReversal serializer" should "work properly" in {
    val gradientReversal = GradientReversal()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = gradientReversal.forward(tensor1)

    ModulePersister.saveToFile("/tmp/gradientReversal.bigdl", gradientReversal, true)
    val loadedGradientReversal = ModuleLoader.loadFromFile("/tmp/gradientReversal.bigdl")
    val res2 = loadedGradientReversal.forward(tensor2)
    res1 should be (res2)
  }

  "Graph serializer " should "work properly" in {
    val linear = Linear(10, 2).inputs()
    val graph = Graph(linear, linear)
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = graph.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/graph.bigdl", graph, true)
    val loadedGraph = ModuleLoader.loadFromFile("/tmp/graph.bigdl")
    val res2 = loadedGraph.forward(tensor2)
    res1 should be (res2)
  }

  "Graph with variables serializer " should "work properly" in {
    val linear = Linear(2, 2)
    val linearNode = linear.inputs()
    val linearWeight = linear.weight
    val linearBias = linear.bias
    val variables = Some(Array(linearWeight), Array(linearBias))
    val graph = Graph(Array(linearNode), Array(linearNode), variables)
    val tensor1 = Tensor(2).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = graph.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/graph.bigdl", graph, true)
    val loadedGraph = ModuleLoader.loadFromFile("/tmp/graph.bigdl")
    val res2 = loadedGraph.forward(tensor2)
    res1 should be (res2)
  }

  "GRU serializer " should "work properly" in {
    RNG.setSeed(100)
    val gru = Recurrent().add(GRU(100, 100))
    val input1 = Tensor(2, 20, 100).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 20, 100)
    input2.copy(input1)

    RNG.setSeed(100)
    val res1 = gru.forward(input1)
    ModulePersister.saveToFile("/tmp/gru.bigdl", gru, true)
    RNG.setSeed(100)
    val loadedGRU = ModuleLoader.loadFromFile("/tmp/gru.bigdl")
    RNG.setSeed(100)
    val res2 = loadedGRU.forward(input2)
    res1 should be (res2)
  }

  "HardShrink serializer" should "work properly" in {
    val hardShrink = HardShrink()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = hardShrink.forward(tensor1)

    ModulePersister.saveToFile("/tmp/hardShrink.bigdl", hardShrink, true)
    val loadedHardShrink = ModuleLoader.loadFromFile("/tmp/hardShrink.bigdl")
    val res2 = loadedHardShrink.forward(tensor2)
    res1 should be (res2)
  }

  "HardTanh serializer" should "work properly" in {
    val hardTanh = HardTanh()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = hardTanh.forward(tensor1)

    ModulePersister.saveToFile("/tmp/hardTanh.bigdl", hardTanh, true)
    val loadedHardTanh = ModuleLoader.loadFromFile("/tmp/hardTanh.bigdl")
    val res2 = loadedHardTanh.forward(tensor2)
    res1 should be (res2)
  }

  "Identity serializer" should "work properly" in {
    val identity = Identity()

    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = identity.forward(tensor1)

    ModulePersister.saveToFile("/tmp/identity.bigdl", identity, true)
    val loadedIdentity = ModuleLoader.loadFromFile("/tmp/identity.bigdl")
    val res2 = loadedIdentity.forward(tensor2)
    res1 should be (res2)
  }

  "Index serializer" should "work properly" in {
    val index = Index(1)

    val input1 = Tensor(3).apply1(e => Random.nextFloat())
    val input2 = Tensor(4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 2
    input2(Array(4)) = 3
    val gradOutput = Tensor(4).apply1(e => Random.nextFloat())

    val input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = index.forward(input)

    ModulePersister.saveToFile("/tmp/index.bigdl", index, true)
    val loadedIndex = ModuleLoader.loadFromFile("/tmp/index.bigdl")
    val res2 = loadedIndex.forward(input)
    res1 should be (res2)
  }

  "InferReshape serializer" should " work properly" in {
    val inferReshape = InferReshape(Array(-1, 2, 0, 5))
    val tensor1 = Tensor(2, 5, 2, 2).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = inferReshape.forward(tensor1)

    ModulePersister.saveToFile("/tmp/inferReshape.bigdl", inferReshape, true)
    val loadedInferReshape = ModuleLoader.loadFromFile("/tmp/inferReshape.bigdl")
    val res2 = loadedInferReshape.forward(tensor2)
    res1 should be (res2)
  }

  "Input serializer " should " work properly " in {
    val input = Input().element
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor(10)
    tensor2.copy(tensor1)

    val res1 = input.forward(tensor1)

    ModulePersister.saveToFile("/tmp/input.bigdl", input, true)
    val loadedInferInput = ModuleLoader.loadFromFile("/tmp/input.bigdl")
    val res2 = loadedInferInput.forward(tensor2)
    res1 should be (res2)
  }

  "JoinTable serializer " should "work  properly" in {
    val joinTable = JoinTable(2, 2)
    val input1 = Tensor(2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 2).apply1(_ => Random.nextFloat())

    val input = T()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val res1 = joinTable.forward(input)

    ModulePersister.saveToFile("/tmp/joinTable.bigdl", joinTable, true)
    val loadedJoinTable = ModuleLoader.loadFromFile("/tmp/joinTable.bigdl")
    val res2 = loadedJoinTable.forward(input)
    res1 should be (res2)

  }

  "L1Penalty serializer " should " work properly" in {
    val l1Penalty = L1Penalty(1, true, true)

    val tensor1 = Tensor(3, 3).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = l1Penalty.forward(tensor1)

    ModulePersister.saveToFile("/tmp/l1Penalty.bigdl", l1Penalty, true)

    val loadedL1Penalty = ModuleLoader.loadFromFile("/tmp/l1Penalty.bigdl")

    val res2 = loadedL1Penalty.forward(tensor2)
    res1 should be (res2)
  }

  "LeakReLu serializer " should " work properly" in {
    val leakyReLU = LeakyReLU(0.01, true)

    val tensor1 = Tensor(3, 3).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = leakyReLU.forward(tensor1)

    ModulePersister.saveToFile("/tmp/leakyReLU.bigdl", leakyReLU, true)

    val loadedLeakyReLU = ModuleLoader.loadFromFile("/tmp/leakyReLU.bigdl")

    val res2 = loadedLeakyReLU.forward(tensor2)
    res1 should be (res2)
  }

  "Linear serializer " should "work properly" in {
    val linear = Linear(10, 2)
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = linear.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/linear.bigdl", linear, true)
    val loadedLinear = ModuleLoader.loadFromFile("/tmp/linear.bigdl")
    val res2 = loadedLinear.forward(tensor2)
    res1 should be (res2)
  }

  "Log Serializer " should " work properly" in {
    val log = Log()
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = log.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/log.bigdl", log, true)
    val loadedLog = ModuleLoader.loadFromFile("/tmp/log.bigdl")
    val res2 = loadedLog.forward(tensor2)
    res1 should be (res2)
  }

  "LogSigmoid serializer" should " work properly" in {
    val logSigmoid = LogSigmoid()
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = logSigmoid.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/logSigmoid.bigdl", logSigmoid, true)
    val loadedLogSigmoid = ModuleLoader.loadFromFile("/tmp/logSigmoid.bigdl")
    val res2 = loadedLogSigmoid.forward(tensor2)
    res1 should be (res2)
  }

  "LogSogMax serializer" should " work properly" in {
    val logSigmoid = LogSoftMax()
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    val res1 = logSigmoid.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModulePersister.saveToFile("/tmp/logSigmoid.bigdl", logSigmoid, true)
    val loadedLogSigmoid = ModuleLoader.loadFromFile("/tmp/logSigmoid.bigdl")
    val res2 = loadedLogSigmoid.forward(tensor2)
    res1 should be (res2)
  }

  "LookupTable serializer " should " work properly" in {
    val lookupTable = LookupTable(9, 4, 2, 0.1, 2.0, true)
    val tensor1 = Tensor(5)
    tensor1(Array(1)) = 5
    tensor1(Array(2)) = 2
    tensor1(Array(3)) = 6
    tensor1(Array(4)) = 9
    tensor1(Array(5)) = 4

    val tensor2 = Tensor(5)
    tensor2.copy(tensor1)

    val res1 = lookupTable.forward(tensor1)

    ModulePersister.saveToFile("/tmp/lookupTable.bigdl", lookupTable, true)
    val loadedLookupTable = ModuleLoader.loadFromFile("/tmp/lookupTable.bigdl")
    val res2 = loadedLookupTable.forward(tensor2)
    res1 should be (res2)
  }

  "LSTM serializer " should " work properly" in {

    val lstm = Recurrent().add(LSTM(6, 4))

    val input1 = Tensor(Array(1, 5, 6)).apply1(_ => Random.nextFloat())
    val input2 = Tensor(1, 5, 6)
    input2.copy(input1)

    val res1 = lstm.forward(input1)

    ModulePersister.saveToFile("/tmp/lstm.bigdl", lstm, true)
    val loadedLSTM = ModuleLoader.loadFromFile("/tmp/lstm.bigdl")
    val res2 = loadedLSTM.forward(input1)
    res1 should be (res2)

  }

  "LSTMPeephole serializer " should " work properly" in {
    val lstmPeephole = Recurrent().add(LSTMPeephole(6, 4))

    val input1 = Tensor(Array(1, 5, 6)).apply1(_ => Random.nextFloat())
    val input2 = Tensor(1, 5, 6)
    input2.copy(input1)

    val res1 = lstmPeephole.forward(input1)

    ModulePersister.saveToFile("/tmp/lstmPeephole.bigdl", lstmPeephole, true)
    val loadedLSTMPeephole = ModuleLoader.loadFromFile("/tmp/lstmPeephole.bigdl")
    val res2 = loadedLSTMPeephole.forward(input2)
    res1 should be (res2)

  }

  "MapTable serializer " should " work properly" in {
    val linear = Linear(2, 2)
    val mapTable = new MapTable()
    mapTable.add(linear)
    val input1 = Tensor(2).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.0.toFloat) = input1
    input(2.0.toFloat) = input2

    val res1 = mapTable.forward(input)

    ModulePersister.saveToFile("/tmp/mapTable.bigdl", mapTable, true)
    val loadedMapTable = ModuleLoader.loadFromFile("/tmp/mapTable.bigdl")
    val res2 = loadedMapTable.forward(input)
    res1 should be (res2)
  }

  "MaskedSelect serializer" should " work properly" in {
    val maskedSelect = MaskedSelect()
    val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 2)
    input2(Array(1, 1)) = 1
    input2(Array(1, 2)) = 0
    input2(Array(2, 1)) = 0
    input2(Array(2, 2)) = 1
    val gradOutput = Tensor(5).apply1(e => Random.nextFloat())

    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2

    val res1 = maskedSelect.forward(input)

    val gradInput = maskedSelect.backward(input, gradOutput)

    ModulePersister.saveToFile("/tmp/maskedSelect.bigdl", maskedSelect, true)
    val loadedMaskedSelect = ModuleLoader.loadFromFile("/tmp/maskedSelect.bigdl")
    val res2 = loadedMaskedSelect.forward(input)
    res1 should be (res2)

  }

  "Max serializer " should " work properly" in {
    val max = new Max(2)
    val input1 = Tensor(2, 3, 4).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)

    val res1 = max.forward(input1)

    ModulePersister.saveToFile("/tmp/max.bigdl", max, true)
    val loadedMax = ModuleLoader.loadFromFile("/tmp/max.bigdl")
    val res2 = loadedMax.forward(input2)
    res1 should be (res2)
  }

  "Mean serializer " should " work properly " in {
    val mean = Mean(2)
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = mean.forward(input1)

    ModulePersister.saveToFile("/tmp/mean.bigdl", mean, true)
    val loadedMean = ModuleLoader.loadFromFile("/tmp/mean.bigdl")
    val res2 = loadedMean.forward(input2)
    res1 should be (res2)
  }

  "Min serializer " should " work properly " in {
    val min = Min(2)
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = min.forward(input1)

    ModulePersister.saveToFile("/tmp/min.bigdl", min, true)
    val loadedMin = ModuleLoader.loadFromFile("/tmp/min.bigdl")
    val res2 = loadedMin.forward(input2)
    res1 should be (res2)
  }

  "MixtureTable Serializer " should " work properly " in {
    val mixTureTable = MixtureTable()
    val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 2).apply1(e => Random.nextFloat())

    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2

    val res1 = mixTureTable.forward(input)

    ModulePersister.saveToFile("/tmp/mixTureTable.bigdl", mixTureTable, true)
    val loadedMixtureTable = ModuleLoader.loadFromFile("/tmp/mixTureTable.bigdl")
    val res2 = loadedMixtureTable.forward(input)
    res1 should be (res2)
  }

  "MM Serializer" should "work properly" in {
    val mm = MM()

    val input1 = Tensor(2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor(3, 4).apply1(e => Random.nextFloat())

    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2

    val res1 = mm.forward(input)

    ModulePersister.saveToFile("/tmp/mm.bigdl", mm, true)
    val loadedMM = ModuleLoader.loadFromFile("/tmp/mm.bigdl")
    val res2 = loadedMM.forward(input)
    res1 should be (res2)

  }

  "Mul Serializer " should "work properly" in {
    val mul = Mul()
    val input1 = Tensor(10, 10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(10, 10)
    input2.copy(input1)

    val res1 = mul.forward(input1)

    ModulePersister.saveToFile("/tmp/mul.bigdl", mul, true)
    val loadedMul = ModuleLoader.loadFromFile("/tmp/mul.bigdl")
    val res2 = loadedMul.forward(input2)
    res1 should be (res2)
  }

  "MulConst Serializer " should "work properly" in {
    val mulConst = MulConstant(1.0)
    val input1 = Tensor(10, 10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(10, 10)
    input2.copy(input1)

    val res1 = mulConst.forward(input1)

    ModulePersister.saveToFile("/tmp/mulConst.bigdl", mulConst, true)
    val loadedMulConstant = ModuleLoader.loadFromFile("/tmp/mulConst.bigdl")
    val res2 = loadedMulConstant.forward(input2)
    res1 should be (res2)
  }

  "MV Serializer " should " work properly" in {
    val mv = MV()
    val input1 = Tensor(2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor(3).apply1(e => Random.nextFloat())

    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2

    val res1 = mv.forward(input)

    ModulePersister.saveToFile("/tmp/mv.bigdl", mv, true)
    val loadedMV = ModuleLoader.loadFromFile("/tmp/mv.bigdl")
    val res2 = loadedMV.forward(input)
    res1 should be (res2)
  }

  "Narrow serializer " should " work properly" in {
    val narrow = Narrow(1, 3, -3)
    val input1 = Tensor(9, 4, 14).apply1(e => Random.nextFloat())
    val input2 = Tensor(9, 4, 14)
    input2.copy(input1)

    val res1 = narrow.forward(input1)

    ModulePersister.saveToFile("/tmp/narrow.bigdl", narrow, true)
    val loadedNarrow = ModuleLoader.loadFromFile("/tmp/narrow.bigdl")
    val res2 = loadedNarrow.forward(input2)
    res1 should be (res2)
  }

  "NarrowTable serializer " should " work properly" in {
    val narrowTable = NarrowTable(1, 1)
    val input = T()
    input(1.0) = Tensor(2, 2).apply1(e => Random.nextFloat())
    input(2.0) = Tensor(2, 2).apply1(e => Random.nextFloat())
    input(3.0) = Tensor(2, 2).apply1(e => Random.nextFloat())
    val res1 = narrowTable.forward(input)

    ModulePersister.saveToFile("/tmp/narrowTable.bigdl", narrowTable, true)
    val loadedNarrowTable = ModuleLoader.loadFromFile("/tmp/narrowTable.bigdl")
    val res2 = loadedNarrowTable.forward(input)
    res1 should be (res2)
  }

  "Normlize serializer " should " work properly" in {
    val normalizer = Normalize(2)
    val input1 = Tensor(2, 3, 4, 4).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 3, 4, 4)
    input2.copy(input1)

    val res1 = normalizer.forward(input1)

    ModulePersister.saveToFile("/tmp/normalizer.bigdl", normalizer, true)
    val loadedNormalize = ModuleLoader.loadFromFile("/tmp/normalizer.bigdl")
    val res2 = loadedNormalize.forward(input2)
    res1 should be (res2)
  }

  "Pack serializer " should " work properly" in {
    val pack = new Pack(1)
    val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input = T()
    input(1.0f) = input1
    input(2.0f) = input2
    val res1 = pack.forward(input)
    ModulePersister.saveToFile("/tmp/pack.bigdl", pack, true)
    val loadedPack = ModuleLoader.loadFromFile("/tmp/pack.bigdl")
    val res2 = loadedPack.forward(input)
    res1 should be (res2)

  }

  "Padding serializer " should " work properly" in {
    val padding = Padding(1, -1, 4, -0.8999761, 14)
    val input = Tensor(3, 13, 11).apply1(e => Random.nextFloat())
    val res1 = padding.forward(input)
    ModulePersister.saveToFile("/tmp/padding.bigdl", padding, true)
    val loadedPadding = ModuleLoader.loadFromFile("/tmp/padding.bigdl")
    val res2 = loadedPadding.forward(input)
    res1 should be (res2)
  }

  "PairwiseDistance serializer " should " work properly" in {
    val pairwiseDistance = new PairwiseDistance(3)
    val input1 = Tensor(3, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor(3, 3).apply1(e => Random.nextFloat())
    val input = T(1.0f -> input1, 2.0f -> input2)
    val res1 = pairwiseDistance.forward(input)
    ModulePersister.saveToFile("/tmp/pairwiseDistance.bigdl", pairwiseDistance, true)
    val loadedPairwiseDistance = ModuleLoader.loadFromFile("/tmp/pairwiseDistance.bigdl")
    val res2 = loadedPairwiseDistance.forward(input)
    res1 should be (res2)
  }

  "ParallelTable serializer " should " work properly" in {
    val parallelTable = ParallelTable()
    parallelTable.add(Linear(2, 2))
    parallelTable.add(Linear(2, 2))
    val input11 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input21 = Tensor(2, 2)
    input21.copy(input11)
    val input12 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input22 = Tensor(2, 2)
    input22.copy(input12)
    val input1 = T(1.0f -> input11, 2.0f -> input12)
    val input2 = T(1.0f -> input21, 2.0f -> input22)
    val res1 = parallelTable.forward(input1)
    ModulePersister.saveToFile("/tmp/parallelTable.bigdl", parallelTable, true)
    val loadedParallelTable = ModuleLoader.loadFromFile("/tmp/parallelTable.bigdl")
    val res2 = loadedParallelTable.forward(input1)
    res1 should be (res2)
  }

  "Power serializer " should " work properly" in {
    val power = Power(2.0)
    val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor(2, 2)
    input2.copy(input1)

    val res1 = power.forward(input1)

    ModulePersister.saveToFile("/tmp/power.bigdl", power, true)
    val loadedPower = ModuleLoader.loadFromFile("/tmp/power.bigdl")
    val res2 = loadedPower.forward(input1)
    res1 should be (res2)
  }

  "PReLU serializer " should " work properly" in {
    val preLu = PReLU(2)
    val input1 = Tensor(2, 3, 4).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 3, 4)
    input2.copy(input1)
    val res1 = preLu.forward(input1)

    ModulePersister.saveToFile("/tmp/preLu.bigdl", preLu, true)
    val loadedPReLU = ModuleLoader.loadFromFile("/tmp/preLu.bigdl")
    val res2 = loadedPReLU.forward(input1)
    res1 should be (res2)
  }

  "Recurrent serializer " should " work properly" in {
    val recurrent = Recurrent()
      .add(RnnCell(5, 4, Tanh()))
    val input1 = Tensor(Array(10, 5, 5))

    val input2 = Tensor(10, 5, 5)
    input2.copy(input1)
    val res1 = recurrent.forward(input1)

    ModulePersister.saveToFile("/tmp/recurrent.bigdl", recurrent, true)
    val loadedRecurrent = ModuleLoader.loadFromFile("/tmp/recurrent.bigdl")
    val res2 = loadedRecurrent.forward(input1)
    res1 should be (res2)

  }

  "ReLU serializer " should " work properly" in {
    val relu = ReLU()
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor(5, 5)
    input2.copy(input1)
    val res1 = relu.forward(input1)

    ModulePersister.saveToFile("/tmp/relu.bigdl", relu, true)
    val loadedReLU = ModuleLoader.loadFromFile("/tmp/relu.bigdl")
    val res2 = loadedReLU.forward(input1)
    res1 should be (res2)
  }

  "ReLU6 serializer" should " work properly " in {
    val relu6 = ReLU6(false)
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = relu6.forward(input1)

    ModulePersister.saveToFile("/tmp/relu6.bigdl", relu6, true)
    val loadedReLU6 = ModuleLoader.loadFromFile("/tmp/relu6.bigdl")
    val res2 = loadedReLU6.forward(input2)
    res1 should be (res2)
  }

  "Replicate serializer " should " work properly" in {
    val replicate = new Replicate(3)
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(10)
    input2.copy(input1)
    val res1 = replicate.forward(input1)
    ModulePersister.saveToFile("/tmp/replicate.bigdl", replicate, true)
    val loadedReplicate = ModuleLoader.loadFromFile("/tmp/replicate.bigdl")
    val res2 = loadedReplicate.forward(input2)
    res1 should be (res2)
  }

  "Reshape serializer " should " work properly " in {
    val reshape = Reshape(Array(1, 4, 5))
    val input1 = Tensor(2, 2, 5).apply1( _ => Random.nextFloat())
    val input2 = Tensor(2, 2, 5)
    input2.copy(input1)
    val res1 = reshape.forward(input1)
    ModulePersister.saveToFile("/tmp/reshape.bigdl", reshape, true)
    val loadedReshape = ModuleLoader.loadFromFile("/tmp/reshape.bigdl")
    val res2 = loadedReshape.forward(input2)
    res1 should be (res2)
  }

  "Reverse serializer " should " work properly " in {
    val reverse = Reverse()
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(10)
    input2.copy(input1)
    val res1 = reverse.forward(input1)
    ModulePersister.saveToFile("/tmp/reverse.bigdl", reverse, true)
    val loadedReverse = ModuleLoader.loadFromFile("/tmp/reverse.bigdl")
    val res2 = loadedReverse.forward(input2)
    res1 should be (res2)
  }

  "RnnCell serializer " should " work properly " in {

    val rnnCell = RnnCell(6, 4, Sigmoid())

    val input1 = Tensor(Array(1, 4)).apply1(_ => Random.nextFloat())

    val input2 = Tensor(Array(1, 4)).apply1(_ => Random.nextFloat())

    val input = T()
    input(1.0f) = input1
    input(2.0f) = input2
    val res1 = rnnCell.forward(input)

    ModulePersister.saveToFile("/tmp/rnnCell.bigdl", rnnCell, true)
    val loadedRnnCell = ModuleLoader.loadFromFile("/tmp/rnnCell.bigdl")
    val res2 = loadedRnnCell.forward(input)
    res1 should be (res2)
  }

  "RoiPooling serializer " should " work properly " in {
    val input1 = T()
    val input2 = T()
    val input11 = Tensor(1, 1, 2, 2).apply1(_ => Random.nextFloat())
    val input21 = Tensor(1, 1, 2, 2)
    input21.copy(input11)
    val input12 = Tensor(1, 5).apply1(_ => Random.nextFloat())
    val input22 = Tensor(1, 5)
    input22.copy(input12)
    input1(1.0f) = input11
    input1(2.0f) = input12
    input2(1.0f) = input21
    input2(2.0f) = input22

    val roiPooling = new RoiPooling[Float](pooledW = 3, pooledH = 2, 1.0f)
    val res1 = roiPooling.forward(input1)
    val res3 = roiPooling.forward(input1)
    ModulePersister.saveToFile("/tmp/roiPooling.bigdl", roiPooling, true)
    val loadedRoiPooling = ModuleLoader.loadFromFile("/tmp/roiPooling.bigdl")
    val res2 = loadedRoiPooling.forward(input2)
    res1 should be (res2)
  }

  "RReLU serializer " should " work properly " in {
    val rrelu = new RReLU(inplace = false)
    val input1 = Tensor(2, 2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 2, 2)
    input2.copy(input1)
    val res1 = rrelu.forward(input1)
    ModulePersister.saveToFile("/tmp/rrelu.bigdl", rrelu, true)
    val loadedRReLU = ModuleLoader.loadFromFile("/tmp/rrelu.bigdl")
    val res2 = loadedRReLU.forward(input2)
    res1 should be (res2)
  }

  "Scale serializer " should " work properly " in {
    val scale = Scale(Array(1, 4, 1, 1))
    val input1 = Tensor(1, 4, 5, 6).apply1(_ => Random.nextFloat())
    val input2 = Tensor(1, 4, 5, 6)
    input2.copy(input1)
    val res1 = scale.forward(input1)
    ModulePersister.saveToFile("/tmp/scale.bigdl", scale, true)
    val loadedScale = ModuleLoader.loadFromFile("/tmp/scale.bigdl")
    val res2 = loadedScale.forward(input2)
    res1 should be (res2)

  }

  "Select serializer " should " work properly " in {
    val select = Select(2, 2)
    val input1 = Tensor(5, 5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor(5, 5, 5)
    input2.copy(input1)
    val res1 = select.forward(input1)
    ModulePersister.saveToFile("/tmp/select.bigdl", select, true)
    val loadedSelect = ModuleLoader.loadFromFile("/tmp/select.bigdl")
    val res2 = loadedSelect.forward(input2)
    res1 should be (res2)
  }

  "SelectTable serializer " should " work properly " in {
    val selectTable = SelectTable(2)
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(10).apply1(_ => Random.nextFloat())
    val input3 = Tensor(10).apply1(_ => Random.nextFloat())
    val input = T(1.0 -> input1, 2.0 -> input2, 3.0 -> input3)
    val res1 = selectTable.forward(input)
    ModulePersister.saveToFile("/tmp/selectTable.bigdl", selectTable, true)
    val loadedSelectTable = ModuleLoader.loadFromFile("/tmp/selectTable.bigdl")
    val res2 = loadedSelectTable.forward(input)
    res1 should be (res2)
  }

  "Sequential Container" should "work properly" in {
    val sequential = Sequential()
    val linear = Linear(10, 2)
    sequential.add(linear)
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = sequential.forward(input1)
    ModulePersister.saveToFile("/tmp/sequential.bigdl", sequential, true)
    val loadedSequential = ModuleLoader.loadFromFile("/tmp/sequential.bigdl")
    val res2 = loadedSequential.forward(input2)
    res1 should be (res2)
  }

  "Sigmoid serializer " should " work properly" in {
    val sigmoid = Sigmoid()
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = sigmoid.forward(input1)
    ModulePersister.saveToFile("/tmp/sigmoid.bigdl", sigmoid, true)
    val loadedSigmoid = ModuleLoader.loadFromFile("/tmp/sigmoid.bigdl")
    val res2 = loadedSigmoid.forward(input2)
    res1 should be (res2)
  }

  "SoftMax serializer " should " work properly" in {
    val softMax = SoftMax()
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = softMax.forward(input1)
    ModulePersister.saveToFile("/tmp/softMax.bigdl", softMax, true)
    val loadedSoftMax = ModuleLoader.loadFromFile("/tmp/softMax.bigdl")
    val res2 = loadedSoftMax.forward(input2)
    res1 should be (res2)
  }

  "SoftMin serializer " should " work properly " in {
    val softMin = SoftMin()
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = softMin.forward(input1)
    ModulePersister.saveToFile("/tmp/softMin.bigdl", softMin, true)
    val loadedSoftMin = ModuleLoader.loadFromFile("/tmp/softMin.bigdl")
    val res2 = loadedSoftMin.forward(input2)
    res1 should be (res2)
  }

  "SoftPlus serializer " should " work properly" in {
    val softPlus = SoftPlus()
    val input1 = Tensor(10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = softPlus.forward(input1)
    ModulePersister.saveToFile("/tmp/softPlus.bigdl", softPlus, true)
    val loadedSoftPlus = ModuleLoader.loadFromFile("/tmp/softPlus.bigdl")
    val res2 = loadedSoftPlus.forward(input2)
    res1 should be (res2)
  }

  "SoftShrink serializer " should " work properly" in {
    val softShrink = SoftShrink()
    val input1 = Tensor(10, 10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = softShrink.forward(input1)
    ModulePersister.saveToFile("/tmp/softShrink.bigdl", softShrink, true)
    val loadedSoftShrink = ModuleLoader.loadFromFile("/tmp/softShrink.bigdl")
    val res2 = loadedSoftShrink.forward(input2)
    res1 should be (res2)
  }

  "SoftSign serializer " should "work properly" in {
    val softSign = SoftSign()
    val input1 = Tensor(10, 10).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = softSign.forward(input1)
    ModulePersister.saveToFile("/tmp/softSign.bigdl", softSign, true)
    val loadedSoftSign = ModuleLoader.loadFromFile("/tmp/softSign.bigdl")
    val res2 = loadedSoftSign.forward(input2)
    res1 should be (res2)
  }

  "SpatialAveragePooling serializer " should " work properly " in {
    val spatialAveragePooling = new SpatialAveragePooling(3, 2, 2, 1)
    val input1 = Tensor(1, 4, 3).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialAveragePooling.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialAveragePooling.bigdl", spatialAveragePooling, true)
    val loadedSpatialAveragePooling = ModuleLoader.loadFromFile("/tmp/spatialAveragePooling.bigdl")
    val res2 = loadedSpatialAveragePooling.forward(input2)
    res1 should be (res2)
  }

  "SpatialBatchNormalization serializer " should " work properly " in {
    val spatialBatchNorm = SpatialBatchNormalization(5)
    val input1 = Tensor(2, 5, 4, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialBatchNorm.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialBatchNorm.bigdl", spatialBatchNorm, true)
    val loadedSpatialBatchNorm = ModuleLoader.loadFromFile("/tmp/spatialBatchNorm.bigdl")
    val res2 = loadedSpatialBatchNorm.forward(input2)
    res1 should be (res2)
  }

  "SpatialContrastiveNormalization serializer " should " work properly" in {
    RNG.setSeed(100)
    val spatialContrastiveNorm = new SpatialContrastiveNormalization()
    val input1 = Tensor(1, 5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialContrastiveNorm.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialContrastiveNorm.bigdl", spatialContrastiveNorm, true)
    RNG.setSeed(100)
    val loadedSpatialContrastiveNorm = ModuleLoader.
      loadFromFile("/tmp/spatialContrastiveNorm.bigdl")
    val res2 = loadedSpatialContrastiveNorm.forward(input2)
    res1 should be (res2)
  }

  "SpatialConvolution serializer " should " work properly" in {
    val spatialConvolution = SpatialConvolution(3, 4, 2, 2)
    val input1 = Tensor(1, 3, 5, 5).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialConvolution.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialConvolution.bigdl", spatialConvolution, true)
    val loadedSpatialConvolution = ModuleLoader.loadFromFile("/tmp/spatialConvolution.bigdl")
    val res2 = loadedSpatialConvolution.forward(input2)
    res1 should be (res2)
  }

  "SpatialConvolutionMap serializer" should " work properly" in {
    val spatialConvolutionMap = SpatialConvolutionMap(
      SpatialConvolutionMap.random(1, 1, 1), 2, 2)
    val input1 = Tensor(1, 3, 3).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialConvolutionMap.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialConvolutionMap.bigdl", spatialConvolutionMap, true)
    val loadedSpatialConvolutionMap = ModuleLoader.
      loadFromFile("/tmp/spatialConvolutionMap.bigdl")
    val res2 = loadedSpatialConvolutionMap.forward(input2)
    res1 should be (res2)
  }

  "SpatialCrossMapLRN serializer " should " work properly " in {
    val spatialCrossMapLRN = SpatialCrossMapLRN(5, 0.01, 0.75, 1.0)
    val input1 = Tensor(2, 2, 2, 2).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialCrossMapLRN.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialCrossMapLRN.bigdl", spatialCrossMapLRN, true)
    val loadedSpatialCrossMapLRN = ModuleLoader.loadFromFile("/tmp/spatialCrossMapLRN.bigdl")
    val res2 = loadedSpatialCrossMapLRN.forward(input2)
    res1 should be (res2)
  }

  "SpatialDilatedConvolution serializer " should "work properly" in {

    val spatialDilatedConvolution = SpatialDilatedConvolution(1, 1,
      2, 2, 1, 1, 0, 0)
    val input1 = Tensor(1, 3, 3).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialDilatedConvolution.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialDilatedConvolution.bigdl",
      spatialDilatedConvolution, true)
    val loadedSpatialDilatedConvolution = ModuleLoader.
      loadFromFile("/tmp/spatialDilatedConvolution.bigdl")
    val res2 = loadedSpatialDilatedConvolution.forward(input2)
    res1 should be (res2)
  }

  "SpatialDivisiveNormalization serializer " should " work properly" in {
    val spatialDivisiveNormalization = SpatialDivisiveNormalization()
    val input1 = Tensor(1, 5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor(1, 5, 5)
    input2.copy(input1)

    val res1 = spatialDivisiveNormalization.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialDivisiveNormalization.bigdl",
      spatialDivisiveNormalization, true)
    val loadedSpatialDivisiveNormalization = ModuleLoader.
      loadFromFile("/tmp/spatialDivisiveNormalization.bigdl")
    val res2 = loadedSpatialDivisiveNormalization.forward(input2)
    res1 should be (res2)

  }

  "SpatialFullConvolution serializer " should " work properly" in {

    val spatialFullConvolution = SpatialFullConvolution(1, 1,
      2, 2, 1, 1, 0, 0)
    val input1 = Tensor(1, 3, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor(1, 3, 3)
    input2.copy(input1)

    val res1 = spatialFullConvolution.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialFullConvolution.bigdl",
      spatialFullConvolution, true)
    val loadedSpatialFullConvolution = ModuleLoader.
      loadFromFile("/tmp/spatialFullConvolution.bigdl")
    val res2 = loadedSpatialFullConvolution.forward(input2)
    res1 should be (res2)
  }

  "SpatialMaxPooling serializer " should " work properly " in {
    val spatialMaxPooling = SpatialMaxPooling(2, 2, 2, 2)
    val input1 = Tensor(1, 3, 3).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialMaxPooling.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialMaxPooling.bigdl",
      spatialMaxPooling, true)
    val loadedSpatialMaxPooling = ModuleLoader.
      loadFromFile("/tmp/spatialMaxPooling.bigdl")
    val res2 = loadedSpatialMaxPooling.forward(input2)
    res1 should be (res2)
  }

  "SpatialShareConvolution serializer " should "work properly" in {
    val spatialShareConvolution = SpatialShareConvolution(1, 1, 2, 2, 1, 1)
    val input1 = Tensor(3, 1, 3, 4).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialShareConvolution.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialShareConvolution.bigdl",
      spatialShareConvolution, true)
    val loadedSpatialShareConvolution = ModuleLoader.
      loadFromFile("/tmp/spatialShareConvolution.bigdl")
    val res2 = loadedSpatialShareConvolution.forward(input2)
    res1 should be (res2)
  }

  "SpatialSubtractiveNormalization serializer " should " work properly" in {
    val kernel = Tensor(3, 3).apply1( e => Random.nextFloat())
    val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)
    val input1 = Tensor(1, 1, 1, 5).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialSubtractiveNormalization.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialSubtractiveNormalization.bigdl",
      spatialSubtractiveNormalization, true)
    val loadedSpatialSubtractiveNormalization = ModuleLoader.
      loadFromFile("/tmp/spatialSubtractiveNormalization.bigdl")
    val res2 = loadedSpatialSubtractiveNormalization.forward(input2)
    res1 should be (res2)
  }

  "SpatialWithinChannelLRN serializer " should " work properly" in {
    val spatialWithinChannelLRN = new SpatialWithinChannelLRN[Float](5, 5e-4, 0.75)
    val input1 = Tensor(1, 4, 7, 6).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = spatialWithinChannelLRN.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialWithinChannelLRN.bigdl",
      spatialWithinChannelLRN, true)
    val loadedSpatialWithinChannelLRN = ModuleLoader.
      loadFromFile("/tmp/spatialWithinChannelLRN.bigdl")
    val res2 = loadedSpatialWithinChannelLRN.forward(input2)
    res1 should be (res2)
  }

  "SpatialZeroPadding serializer " should " work properly" in {
    val spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
    val input1 = Tensor(3, 3, 3).apply1(_ => Random.nextFloat())
    val input2 = Tensor(3, 3, 3)
    input2.copy(input1)
    val res1 = spatialZeroPadding.forward(input1)
    ModulePersister.saveToFile("/tmp/spatialZeroPadding.bigdl",
      spatialZeroPadding, true)
    val loadedSpatialSpatialZeroPadding = ModuleLoader.
      loadFromFile("/tmp/spatialZeroPadding.bigdl")
    val res2 = loadedSpatialSpatialZeroPadding.forward(input2)
    res1 should be (res2)

  }

  "SplitTable serializer " should " work properly" in {
    val splitTable = SplitTable(2)
    val input1 = Tensor(2, 10).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = splitTable.forward(input1)
    ModulePersister.saveToFile("/tmp/splitTable.bigdl", splitTable, true)
    val loadedSplitTable = ModuleLoader.loadFromFile("/tmp/splitTable.bigdl")
    val res2 = loadedSplitTable.forward(input2)
    res1 should be (res2)
  }

  "Sqrt serializer " should " work properly" in {
    val sqrt = Sqrt()
    val input1 = Tensor(10).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = sqrt.forward(input1)
    ModulePersister.saveToFile("/tmp/sqrt.bigdl", sqrt, true)
    val loadedSqrt = ModuleLoader.loadFromFile("/tmp/sqrt.bigdl")
    val res2 = loadedSqrt.forward(input2)
    res1 should be (res2)
  }

  "Square serializer " should " work properly " in {
    val square = Square()
    val input1 = Tensor(10).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = square.forward(input1)
    ModulePersister.saveToFile("/tmp/square.bigdl", square, true)
    val loadedSquare = ModuleLoader.loadFromFile("/tmp/square.bigdl")
    val res2 = loadedSquare.forward(input2)
    res1 should be (res2)
  }

  "Squeeze serializer " should " work properly" in {
    val squeeze = Squeeze(2)
    val input1 = Tensor(2, 1, 2).apply1( e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = squeeze.forward(input1)
    ModulePersister.saveToFile("/tmp/squeeze.bigdl", squeeze, true)
    val loadedSqueeze = ModuleLoader.loadFromFile("/tmp/squeeze.bigdl")
    val res2 = loadedSqueeze.forward(input2)
    res1 should be (res2)
  }

  "Sum serializer " should " work properly" in {
    val sum = Sum(2)
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = sum.forward(input1)

    ModulePersister.saveToFile("/tmp/sum.bigdl", sum, true)
    val loadedSum = ModuleLoader.loadFromFile("/tmp/sum.bigdl")
    val res2 = loadedSum.forward(input2)
    res1 should be (res2)
  }

  "Tanh serializer" should " work properly" in {
    val tanh = Tanh()
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = tanh.forward(input1)

    ModulePersister.saveToFile("/tmp/tanh.bigdl", tanh, true)
    val loadedTanh = ModuleLoader.loadFromFile("/tmp/tanh.bigdl")
    val res2 = loadedTanh.forward(input2)
    res1 should be (res2)
  }

  "TanhShrink serializer " should " work properly" in {
    val tanhShrink = TanhShrink()
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val res1 = tanhShrink.forward(input1)

    ModulePersister.saveToFile("/tmp/tanhShrink.bigdl", tanhShrink, true)
    val loadedTanhShrink = ModuleLoader.loadFromFile("/tmp/tanhShrink.bigdl")
    val res2 = loadedTanhShrink.forward(input2)
    res1 should be (res2)
  }

  "TemporalConvolution serializer " should " work properly" in {
    val temporalConvolution = TemporalConvolution(10, 8, 5, 2)
    val input1 = Tensor(100, 10).apply1(e => Random.nextFloat())
    val input2 = Tensor(100, 10)
    input2.copy(input1)

    val res1 = temporalConvolution.forward(input1)

    ModulePersister.saveToFile("/tmp/temporalConvolution.bigdl", temporalConvolution, true)
    val loadedTemporalConvolution = ModuleLoader.loadFromFile("/tmp/temporalConvolution.bigdl")
    val res2 = loadedTemporalConvolution.forward(input2)
    res1 should be (res2)
  }

  "Threshold serializer " should " work properly" in {
    val threshold = Threshold(0.5)
    val input1 = Tensor(5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor(5, 5)
    input2.copy(input1)
    val res1 = threshold.forward(input1)

    ModulePersister.saveToFile("/tmp/threshold.bigdl", threshold, true)
    val loadedThreshold = ModuleLoader.loadFromFile("/tmp/threshold.bigdl")
    val res2 = loadedThreshold.forward(input1)
    res1 should be (res2)
  }

  "TimeDistributed serializer " should " work properly" in {
    val timeDistributed = TimeDistributed(Linear(5, 5))
    val input1 = Tensor(2, 5, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 5, 5)
    input2.copy(input1)
    val res1 = timeDistributed.forward(input1)

    ModulePersister.saveToFile("/tmp/timeDistributed.bigdl", timeDistributed, true)
    val loadedTimeDistributed = ModuleLoader.loadFromFile("/tmp/timeDistributed.bigdl")
    val res2 = loadedTimeDistributed.forward(input1)
    res1 should be (res2)
  }

  "Transpose serializer " should " work properly" in {
    val transpose = Transpose(Array((1, 2)))
    val input1 = Tensor().resize(Array(2, 3)).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 3)
    input2.copy(input1)

    val res1 = transpose.forward(input1)

    ModulePersister.saveToFile("/tmp/transpose.bigdl", transpose, true)
    val loadedTranspose = ModuleLoader.loadFromFile("/tmp/transpose.bigdl")
    val res2 = loadedTranspose.forward(input1)
    res1 should be (res2)

  }

  "Unsqueeze serializer" should " work properly" in {
    val unsqueeze = Unsqueeze(2)
    val input1 = Tensor(2, 2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 2, 2)
    input2.copy(input1)
    val res1 = unsqueeze.forward(input1)

    ModulePersister.saveToFile("/tmp/unsqueeze.bigdl", unsqueeze, true)
    val loadedUnsqueeze = ModuleLoader.loadFromFile("/tmp/unsqueeze.bigdl")
    val res2 = loadedUnsqueeze.forward(input1)
    res1 should be (res2)
  }

  "View serializer" should " work properly " in {
    val view = View(Array(2, 5))
    val input1 = Tensor(1, 10).apply1(_ => Random.nextFloat())
    val input2 = Tensor(1, 10)
    input2.copy(input1)
    val res1 = view.forward(input1)

    ModulePersister.saveToFile("/tmp/view.bigdl", view, true)
    val loadedView = ModuleLoader.loadFromFile("/tmp/view.bigdl")
    val res2 = loadedView.forward(input1)
    res1 should be (res2)
  }

  "VolumetricConvolution serializer " should " work properly " in {
    val volumetricConvolution = VolumetricConvolution(2, 3, 2, 2, 2, dT = 1, dW = 1, dH = 1,
      padT = 0, padW = 0, padH = 0, withBias = true)
    val input1 = Tensor(2, 2, 2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor(2, 2, 2, 2)
    input2.copy(input1)
    val res1 = volumetricConvolution.forward(input1)

    ModulePersister.saveToFile("/tmp/volumetricConvolution.bigdl", volumetricConvolution, true)
    val loadedVolumetricConvolution = ModuleLoader.loadFromFile("/tmp/volumetricConvolution.bigdl")
    val res2 = loadedVolumetricConvolution.forward(input1)
    res1 should be (res2)
  }

  "VolumetricFullConvolution serializer " should " work properly " in {

    val volumetricFullConvolution = new VolumetricFullConvolution(3, 6,
      4, 3, 3, 2, 1, 1, 2, 2, 2)

    val input1 = Tensor(3, 3, 3, 6, 6).apply1(e => Random.nextFloat())
    val input2 = Tensor(3, 3, 3, 6, 6).copy(input1)

    val res1 = volumetricFullConvolution.forward(input1)

    ModulePersister.saveToFile("/tmp/volumetricFullConvolution.bigdl",
      volumetricFullConvolution, true)
    val loadedVolumetricFullConvolution = ModuleLoader.
      loadFromFile("/tmp/volumetricFullConvolution.bigdl")
    val res2 = loadedVolumetricFullConvolution.forward(input1)
    res1 should be (res2)

  }

  "VolumetricMaxPooling serializer " should " work properly " in {
    val volumetricMaxPooling = VolumetricMaxPooling(2, 2, 2, 1, 1, 1, 0, 0, 0)
    val input1 = Tensor(1, 2, 3, 3).apply1(_ => Random.nextFloat())
    val input2 = Tensor(1, 2, 3, 3)
    input2.copy(input1)
    val res1 = volumetricMaxPooling.forward(input1)

    ModulePersister.saveToFile("/tmp/volumetricMaxPooling.bigdl", volumetricMaxPooling, true)
    val loadedVolumetricMaxPooling = ModuleLoader.loadFromFile("/tmp/volumetricMaxPooling.bigdl")
    val res2 = loadedVolumetricMaxPooling.forward(input1)
    res1 should be (res2)
  }
  "Customized Module " should "work properly" in {
    val testModule = new TestModule(1.0)
    ModuleSerializer.registerModule("com.intel.analytics.bigdl.utils.serializer.TestModule",
    TestSerializer)
    val tensor1 = Tensor(10).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = testModule.forward(tensor1)
    ModulePersister.saveToFile("/tmp/testModule.bigdl", testModule, true)
    val loadedModule = ModuleLoader.loadFromFile("/tmp/testModule.bigdl")
    val res2 = loadedModule.forward(tensor2)
    res1 should be (res2)
  }
}

class TestModule[T: ClassTag](val constant_scalar: Double)
                             (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val addConst = AddConstant(constant_scalar)
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = addConst.forward(input).asInstanceOf[Tensor[T]]
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = addConst.updateGradInput(input, gradOutput).asInstanceOf[Tensor[T]]
    gradInput
  }
}
case object TestSerializer extends ModuleSerializable
