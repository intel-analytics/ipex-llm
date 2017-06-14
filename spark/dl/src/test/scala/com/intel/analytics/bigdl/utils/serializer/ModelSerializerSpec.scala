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
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import org.scalatest.{FlatSpec, Matchers}
import serialization.Model.{BigDLModel, CustomParam}
import serialization.Model.BigDLModel.ModuleType
import serialization.Serialization
import serialization.Serialization.TestParam

import scala.reflect.ClassTag
import scala.util.Random


class ModelSerializerSpec extends FlatSpec with Matchers {

  "Abs serializer" should "work properly" in {
    val abs = Abs[Double]()
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = abs.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/abs.bigdl", abs, true)
    val loadedAbs = ModelLoader.loadFromFile("/tmp/abs.bigdl").asInstanceOf[Abs[Double]]
    val res2 = loadedAbs.forward(tensor2)
    res1 should be (res2)
  }

  "Add serializer" should "work properly" in {
    val add = Add[Double](5)
    val tensor1 = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = add.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/add.bigdl", add, true)
    val loadedAdd = ModelLoader.loadFromFile("/tmp/add.bigdl")
    val res2 = loadedAdd.asInstanceOf[Add[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "AddConst serializer" should "work properly" in {
    val addconst = AddConstant[Double](5)
    val tensor1 = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = addconst.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/addconst.bigdl", addconst, true)
    val loadedAddConst = ModelLoader.loadFromFile("/tmp/addconst.bigdl")
    val res2 = loadedAddConst.asInstanceOf[AddConstant[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "BatchNorm serializer" should "work properly" in {
    val batchNorm = BatchNormalization[Double](5)
    val tensor1 = Tensor[Double](2, 5).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = batchNorm.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/batchNorm.bigdl", batchNorm, true)
    val loadedBatchNorm = ModelLoader.loadFromFile("/tmp/batchNorm.bigdl")
    val res2 = loadedBatchNorm.asInstanceOf[BatchNormalization[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "BiLinear serializer" should "work properly" in {
    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val biLinear = Bilinear[Double](5, 3, 2)
    val res1 = biLinear.forward(input)
    ModelPersister.saveToFile("/tmp/biLinear.bigdl", biLinear, true)
    val loadedBiLinear = ModelLoader.loadFromFile("/tmp/biLinear.bigdl")
    val res2 = loadedBiLinear.asInstanceOf[Bilinear[Double]].forward(input)
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
    val input1 = Tensor[Double](10).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val bottle = new Bottle[Double](Linear[Double](10, 2).asInstanceOf[Module[Double]], 2, 2)

    val res1 = bottle.forward(input1)
    ModelPersister.saveToFile("/tmp/bottle.bigdl", bottle, true)
    val loadedBottle = ModelLoader.loadFromFile("/tmp/bottle.bigdl")
    val res2 = loadedBottle.asInstanceOf[Bottle[Double]].forward(input2)
    res1 should be (res2)
  }

  "Caddserializer" should "work properly" in {
    val input1 = Tensor[Double](5, 1).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val cadd = CAdd[Double](Array(5, 1))

    val res1 = cadd.forward(input1)
    ModelPersister.saveToFile("/tmp/cadd.bigdl", cadd, true)
    val loadedCadd = ModelLoader.loadFromFile("/tmp/cadd.bigdl")
    val res2 = loadedCadd.asInstanceOf[CAdd[Double]].forward(input2)
    res1 should be (res2)
  }

  "CaddTable serializer" should "work properly" in {
    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val caddTable = CAddTable[Double](false)

    val res1 = caddTable.forward(input)
    ModelPersister.saveToFile("/tmp/caddTable.bigdl", caddTable, true)
    val loadedCaddTable = ModelLoader.loadFromFile("/tmp/caddTable.bigdl")
    val res2 = loadedCaddTable.asInstanceOf[CAddTable[Double]].forward(input)
    res1 should be (res2)
  }

  "CDivTable serializer" should "work properly" in {
    val cdivTable = new CDivTable[Double]()
    val input1 = Tensor[Double](10).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](10).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = cdivTable.forward(input)

    ModelPersister.saveToFile("/tmp/cdivTable.bigdl", cdivTable, true)
    val loadedCdivTable = ModelLoader.loadFromFile("/tmp/cdivTable.bigdl")
    val res2 = cdivTable.asInstanceOf[CDivTable[Double]].forward(input)
    res1 should be (res2)
  }

  "Clamp serializer" should "work properly" in {

    val input1 = Tensor[Double](10).apply1(e => Random.nextDouble())

    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val clamp = Clamp[Double](1, 10)
    val res1 = clamp.forward(input1)

    ModelPersister.saveToFile("/tmp/clamp.bigdl", clamp, true)
    val loadedClamp = ModelLoader.loadFromFile("/tmp/clamp.bigdl")
    val res2 = loadedClamp.asInstanceOf[Clamp[Double]].forward(input2)
    res1 should be (res2)
  }

  "CMaxTable serializer" should "work properly" in {
    val cmaxTable = new CMaxTable[Double]()
    val input1 = Tensor[Double](10).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](10).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = cmaxTable.forward(input)

    ModelPersister.saveToFile("/tmp/cmaxTable.bigdl", cmaxTable, true)
    val loadedCmaxTable = ModelLoader.loadFromFile("/tmp/cmaxTable.bigdl")
    val res2 = loadedCmaxTable.asInstanceOf[CMaxTable[Double]].forward(input)
    res1 should be (res2)
  }

  "CMinTable serializer" should "work properly" in {
    val cminTable = new CMinTable[Double]()
    val input1 = Tensor[Double](10).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](10).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = cminTable.forward(input)

    ModelPersister.saveToFile("/tmp/cminTable.bigdl", cminTable, true)
    val loadedCminTable = ModelLoader.loadFromFile("/tmp/cminTable.bigdl")
    val res2 = loadedCminTable.asInstanceOf[CMinTable[Double]].forward(input)
    res1 should be (res2)
  }

  "CMulserializer" should "work properly" in {
    val input1 = Tensor[Double](5, 1).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val cmul = CMul[Double](Array(5, 1))

    val res1 = cmul.forward(input1)
    ModelPersister.saveToFile("/tmp/cmul.bigdl", cmul, true)
    val loadedCmul = ModelLoader.loadFromFile("/tmp/cmul.bigdl")
    val res2 = loadedCmul.asInstanceOf[CMul[Double]].forward(input2)
    res1 should be (res2)
  }

  "CMulTable serializer" should "work properly" in {
    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val cmulTable = CMulTable[Double]()

    val res1 = cmulTable.forward(input)
    ModelPersister.saveToFile("/tmp/cmulTable.bigdl", cmulTable, true)
    val loadedCmulTable = ModelLoader.loadFromFile("/tmp/cmulTable.bigdl")
    val res2 = loadedCmulTable.asInstanceOf[CMulTable[Double]].forward(input)
    res1 should be (res2)
  }

  "Concatserializer" should "work properly" in {
    val input1 = Tensor[Double](2, 2, 2).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val concat = Concat[Double](2)

    concat.add(Abs[Double]())
    concat.add(Abs[Double]())

    val res1 = concat.forward(input1)
    ModelPersister.saveToFile("/tmp/concat.bigdl", concat, true)
    val loadedConcat = ModelLoader.loadFromFile("/tmp/concat.bigdl")
    val res2 = loadedConcat.asInstanceOf[Concat[Double]].forward(input2)
    res1 should be (res2)
  }

  "ConcatTable serializer" should "work properly" in {
    val concatTable = new ConcatTable[Double]()
    concatTable.add(Linear[Double](10, 2))
    concatTable.add(Linear[Double](10, 2))

    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    tensor2.resizeAs(tensor1).copy(tensor1)
    val res1 = concatTable.forward(tensor1)

    ModelPersister.saveToFile("/tmp/concatTable.bigdl", concatTable, true)
    val loadedConcatTable = ModelLoader.loadFromFile("/tmp/concatTable.bigdl")
    val res2 = loadedConcatTable.asInstanceOf[ConcatTable[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "Contiguous serializer" should "work properly" in {
    val contiguous = new Contiguous[Double]()

    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = contiguous.forward(tensor1)

    ModelPersister.saveToFile("/tmp/contiguous.bigdl", contiguous, true)
    val loadedContiguous = ModelLoader.loadFromFile("/tmp/contiguous.bigdl")
    val res2 = loadedContiguous.asInstanceOf[Contiguous[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "Cosine serializer" should "work properly" in {
    val cosine = new Cosine[Double](5, 5)

    val tensor1 = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    tensor2.resizeAs(tensor1).copy(tensor1)

    val res1 = cosine.forward(tensor1)

    ModelPersister.saveToFile("/tmp/cosine.bigdl", cosine, true)
    val loadedCosine = ModelLoader.loadFromFile("/tmp/cosine.bigdl")
    val res2 = loadedCosine.asInstanceOf[Cosine[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "CosineDistance serializer" should "work properly" in {
    val cosineDistance = CosineDistance[Double]()

    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = cosineDistance.forward(input)

    ModelPersister.saveToFile("/tmp/cosineDistance.bigdl", cosineDistance, true)
    val loadedCosineDistance = ModelLoader.loadFromFile("/tmp/cosineDistance.bigdl")
    val res2 = loadedCosineDistance.asInstanceOf[CosineDistance[Double]].forward(input)
    res1 should be (res2)
  }

  "CSubTable serializer" should "work properly" in {
    val csubTable = CSubTable[Double]()

    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = csubTable.forward(input)

    ModelPersister.saveToFile("/tmp/csubTable.bigdl", csubTable, true)
    val loadedCSubTable = ModelLoader.loadFromFile("/tmp/csubTable.bigdl")
    val res2 = loadedCSubTable.asInstanceOf[CSubTable[Double]].forward(input)
    res1 should be (res2)
  }

  "Dotproduct serializer" should "work properly" in {
    val dotProduct = DotProduct[Double]()

    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val res1 = dotProduct.forward(input)

    ModelPersister.saveToFile("/tmp/dotProduct.bigdl", dotProduct, true)
    val loadedDotProduct = ModelLoader.loadFromFile("/tmp/dotProduct.bigdl")
    val res2 = loadedDotProduct.asInstanceOf[DotProduct[Double]].forward(input)
    res1 should be (res2)
  }

  "Linear serializer" should "work properly" in {
    val linear = Linear[Double](10, 2)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = linear.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/linear.bigdl", linear, true)
    val loadedLinear = ModelLoader.loadFromFile("/tmp/linear.bigdl")
    val res2 = loadedLinear.asInstanceOf[Linear[Double]].forward(tensor2)
    res1 should be (res2)
   }

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

  "Graph Container" should "work properly" in {
    val linear = Linear[Double](10, 2).apply()
    val graph = Graph(linear, linear)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = graph.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelPersister.saveToFile("/tmp/graph.bigdl", graph, true)
    val loadedGraph = ModelLoader.loadFromFile("/tmp/graph.bigdl")
    val res2 = loadedGraph.asInstanceOf[Graph[Double]].forward(tensor2)
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
class TestModule[T: ClassTag](override val constant_scalar: Double)
  (implicit ev: TensorNumeric[T]) extends AddConstant[T](constant_scalar) {
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
}
