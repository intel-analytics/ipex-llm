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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.serializer._
import org.scalatest.{FlatSpec, Matchers}
import scala.reflect.ClassTag
import scala.util.Random


class ModelSerializerSpec extends FlatSpec with Matchers {

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
    ModulePersister.saveModelDefinitionToFile("/tmp/biLinear.prototxt", biLinear, true)
    val loadedBiLinear = ModuleLoader.loadFromFile("/tmp/biLinear.bigdl")
    val res2 = loadedBiLinear.forward(input)
    res1 should be (res2)
  }
/*
  "BiRecurrent serializer" should "work properly" in {
    val input1 = Tensor[Double](5, 5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)
    val biRecurrent = BiRecurrent[Double]()
    val res1 = biRecurrent.forward(input1)
    ModelPersister.saveToFile("/tmp/biRecurrent.bigdl", biRecurrent, true)
    val loadedRecurent = ModelLoader.loadFromFile("/tmp/biRecurrent.bigdl")
    val res2 = loadedRecurent.asInstanceOf[BiRecurrent[Double]].forward(input2)
    res1 should be (res2)
  }
  */

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

  "Cell serializer" should "work properly" in {

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
    ModulePersister.saveModelDefinitionToFile("/tmp/concatTable.prototxt", concatTable, true)
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
    ModulePersister.saveModelDefinitionToFile("/tmp/graph.prototxt", graph, true)
    val loadedGraph = ModuleLoader.loadFromFile("/tmp/graph.bigdl")
    val res2 = loadedGraph.forward(tensor2)
    res1 should be (res2)
  }
/*
  "GRU serializer " should "work properly" in {
    val gru = GRU(10, 10)
    val input1 = Tensor(10, 10).apply1(e => Random.nextFloat())
    val input2 = Tensor(10, 10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2
    val res1 = gru.forward(input)
    ModulePersister.saveToFile("/tmp/gru.bigdl", gru, true)
    val loadedGRU = ModuleLoader.loadFromFile("/tmp/gru.bigdl")
    val res2 = loadedGRU.forward(input)
    res1 should be (res2)
  }
*/
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

  }

/*
  "Sequantial Container" should "work properly" in {
    val sequential = Sequential[Double]()
    val linear = Linear[Double](10, 2)
    sequential.add(linear)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = sequential.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/sequential.bigdl", sequential, true)
    val loadedSequential = ModelLoader.loadFromFile("/tmp/sequential.bigdl")
    val res2 = loadedSequential.asInstanceOf[Sequential[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "Customized Module " should "work properly" in {
    val testModule = new TestModule[Double](1.0)
    CustomizedDelegator.registerCustomizedModule(testModule.getClass,
      TestSerializer, Serialization.customizedData, "Test")
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = testModule.forward(tensor1)
    ModelPersister.saveToFile("/tmp/testModule.bigdl", testModule, true)
    ModelPersister.saveModelDefinitionToFile("/tmp/testModule.prototxt", testModule, true)
    val loadedModule = ModelLoader.loadFromFile("/tmp/testModule.bigdl")
    val res2 = loadedModule.asInstanceOf[TestModule[Double]].forward(tensor2)
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
case object TestSerializer extends AbstractModelSerializer {

  override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
  : BigDLModule[T] = {
    val customParam = model.getCustomParam
    val customType = customParam.getCustomType
    val customizedData = customParam.getExtension(Serialization.customizedData)
    createBigDLModule(model, new TestModule(customizedData.getScalar).
      asInstanceOf[AbstractModule[Activity, Activity, T]])
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]): BigDLModel = {
    val bigDLModelBuilder = BigDLModel.newBuilder
    bigDLModelBuilder.setModuleType(ModuleType.CUSTOMIZED)
    val customParam = CustomParam.newBuilder
    customParam.setCustomType("Test")
    val testParam = TestParam.newBuilder
    testParam.setScalar(module.module.asInstanceOf[TestModule[T]].constant_scalar)
    customParam.setExtension(Serialization.customizedData, testParam.build)
    bigDLModelBuilder.setCustomParam(customParam.build)
    createSerializeBigDLModule(bigDLModelBuilder, module)
  }
  */
}
