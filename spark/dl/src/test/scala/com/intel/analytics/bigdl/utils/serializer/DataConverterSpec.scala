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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.VariableFormat.{Default, ONE_D}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat.{NCHW, NHWC}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.quantized.{LinearWeight, LinearWeightParams}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, Storage, Tensor}
import com.intel.analytics.bigdl.utils.{MultiShape, SingleShape, Shape => BigDLShape}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLTensor, DataType, TensorStorage}

import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.SingleShape
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue.ArrayValue

import scala.collection.mutable
import scala.util.Random


class DataConverterSpec extends FlatSpec with Matchers{

  val map = new mutable.HashMap[Int, Any]()

  "Primitive Int type conversion" should "work properly" in {
    val intValue = 1
    val attriBulder = AttrValue.newBuilder
    val intType = universe.typeOf[Int]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, intValue, intType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue should be (intValue)
  }

  "Primitive Long type conversion" should "work properly" in {
    val longValue = 1L
    val attriBulder = AttrValue.newBuilder
    val longType = universe.typeOf[Long]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, longValue, longType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null)
      , attriBulder.build)
    retrievedValue should be (longValue)
  }

  "Primitive Float type conversion" should "work properly" in {
    val floatValue = 1.0f
    val attriBulder = AttrValue.newBuilder
    val floatType = universe.typeOf[Float]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, floatValue, floatType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null)
      , attriBulder.build)
    retrievedValue should be (floatValue)
  }

  "Primitive Double type conversion" should "work properly" in {
    val doubleValue = 1.0
    val attriBulder = AttrValue.newBuilder
    val doubleType = universe.typeOf[Double]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, doubleValue, doubleType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue should be (doubleValue)
  }

  "Primitive String type conversion" should "work properly" in {
    val strValue = "test"
    val attriBulder = AttrValue.newBuilder
    val strType = universe.typeOf[String]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, strValue, strType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue should be (strValue)
  }

  "Primitive Boolean type conversion" should  "work properly" in {
    val boolValue = false
    val attriBulder = AttrValue.newBuilder
    val boolType = universe.typeOf[Boolean]
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, boolValue, boolType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue.isInstanceOf[Boolean] should be (true)
    retrievedValue.asInstanceOf[Boolean] should be (boolValue)
  }

  "L1L2Regularizer conversion" should  "work properly" in {
    val regularizer = L1L2Regularizer(1.0, 2.0)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, regularizer, ModuleSerializer.regularizerType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue.isInstanceOf[L1L2Regularizer[_]] should be (true)
    retrievedValue.asInstanceOf[L1L2Regularizer[_]].l1 should be (regularizer.l1)
    retrievedValue.asInstanceOf[L1L2Regularizer[_]].l2 should be (regularizer.l2)
  }

  "L1Regularizer conversion" should  "work properly" in {
    val regularizer = L1Regularizer(1.0)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, regularizer, ModuleSerializer.regularizerType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue.isInstanceOf[L1Regularizer[_]] should be (true)
    retrievedValue.asInstanceOf[L1Regularizer[Float]].l1 should be (regularizer.l1)
  }

  "L2Regularizer conversion" should  "work properly" in {
    val regularizer = L2Regularizer(1.0)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, regularizer, ModuleSerializer.regularizerType)
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attriBulder.build)
    retrievedValue.isInstanceOf[L2Regularizer[_]] should be (true)
    retrievedValue.asInstanceOf[L2Regularizer[Float]].l2 should be (regularizer.l2)
  }

  "Empty Regularizer conversion" should  "work properly" in {
    val regularizer : L1L2Regularizer[Float] = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, regularizer, ModuleSerializer.regularizerType)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map, null),
      attr)
    attr.getDataType should be (DataType.REGULARIZER)
    retrievedValue should be (regularizer)
  }

  "Tensor conversion" should "work properly" in {
    val tensor = Tensor(5, 5).apply1(e => Random.nextFloat())
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, tensor, ModuleSerializer.tensorType)
    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(tensor)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(tensor.storage.array())).get.asInstanceOf[TensorStorage])
    attriBulder.setTensorValue(tensorBuilder.build)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.TENSOR)
    retrievedValue should be (tensor)
  }

  "Null Tensor conversion" should "work properly" in {
    val tensor : Tensor[Float] = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, tensor, ModuleSerializer.tensorType)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.TENSOR)
    retrievedValue should be (tensor)
  }

  "Empty Tensor conversion" should "work properly" in {
    val tensor : Tensor[Float] = Tensor[Float]()
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, tensor, ModuleSerializer.tensorType)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.TENSOR)
    retrievedValue should be (tensor)
  }

  "Two tensors to the same object conversion" should "work properly" in {
    val tensor1 = Tensor(5, 5).apply1(e => Random.nextFloat())
    val tensor2 = tensor1

    map.clear()

    val attriBulder1 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder1, tensor1, ModuleSerializer.tensorType)

    val tensorBuilder1 = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(tensor1)).get.asInstanceOf[BigDLTensor])
    tensorBuilder1.setStorage(map.get(System.
      identityHashCode(tensor1.storage.array())).get.asInstanceOf[TensorStorage])
    attriBulder1.setTensorValue(tensorBuilder1.build)
    val attr1 = attriBulder1.build


    val attriBulder2 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder2, tensor2, ModuleSerializer.tensorType)
    val tensorBuilder2 = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(tensor2)).get.asInstanceOf[BigDLTensor])
    tensorBuilder2.setStorage(map.get(System.
      identityHashCode(tensor2.storage.array())).get.asInstanceOf[TensorStorage])
    attriBulder2.setTensorValue(tensorBuilder2.build)
    val attr2 = attriBulder2.build

    map.clear()

    val retrievedValue1 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr1)

    val retrievedValue2 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr2)

    retrievedValue1.asInstanceOf[Tensor[Float]].resize(1, 25)

    retrievedValue2 should be (retrievedValue1)
  }

  "Two tensor share the same memory" should "work properly" in {
    val array = Array[Float](1.0f, 2.0f, 3.0f, 4.0f)
    val storage = Storage[Float](array)
    val tensor1 = Tensor(storage, 1)
    val tensor2 = Tensor(storage, 1)

    map.clear()

    val attriBulder1 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder1, tensor1, ModuleSerializer.tensorType)

    val attr1 = attriBulder1.build

    val attriBulder2 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder2, tensor2, ModuleSerializer.tensorType)
    val attr2 = attriBulder2.build

    map.clear()

    val retrievedValue1 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr1)
      .asInstanceOf[Tensor[Float]]

    retrievedValue1.storage().array()(0) = 10.0f

    val retrievedValue2 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr2)
      .asInstanceOf[Tensor[Float]]

    retrievedValue1.storage() should be (retrievedValue2.storage())
  }

  "Two tensors share the same storage" should "work properly" in {
    val weight = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)

    val storage = Storage(weight)
    val tensor1 = Tensor(Storage(weight), 1, Array(2, 2))
    val tensor2 = Tensor(Storage(weight), 5, Array(2, 2))

    map.clear()

    val attriBulder1 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder1, tensor1, ModuleSerializer.tensorType)

    val attr1 = attriBulder1.build

    val attriBulder2 = AttrValue.newBuilder

    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder2, tensor2, ModuleSerializer.tensorType)
    val attr2 = attriBulder2.build

    map.clear()

    val retrievedValue1 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr1)
      .asInstanceOf[Tensor[Float]]

    val retrievedValue2 = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr2)
      .asInstanceOf[Tensor[Float]]

    retrievedValue1.storage().array().update(1, 0.1f)

    retrievedValue1.storage() should be (retrievedValue2.storage())

  }


  "VariableFormat conversion " should " work properly" in {
    val format : VariableFormat = Default
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }

  "VariableFormat conversion With Param " should " work properly" in {
    val format : VariableFormat = ONE_D
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }

  "Empty VariableFormat conversion " should " work properly" in {
    val format : VariableFormat = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }

  "Init Method conversion " should " work properly" in {
    val initMethod = RandomUniform
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, initMethod, universe.typeOf[InitializationMethod])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.INITMETHOD)
    retrievedValue should be (initMethod)
  }

  "Empty Init Method conversion " should " work properly" in {
    val initMethod : InitializationMethod = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, initMethod, universe.typeOf[InitializationMethod])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.INITMETHOD)
    retrievedValue should be (initMethod)
  }


  "Module Conversion" should "work properly" in {
    val linear = Linear(5, 5).setName("linear")
    val moduleData = ModuleData(linear, Seq(), Seq())
    map.clear()
    ModulePersister.saveToFile("/tmp/linear.bigdl", null, linear, true)
    map.clear()
    val retrievedValue = ModuleLoader.loadFromFile("/tmp/linear.bigdl")
    retrievedValue should be (linear)
  }


  "Nullable Module Conversion" should "work properly" in {
    val linear : TensorModule[Float] = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, linear, ModuleSerializer.abstractModuleType)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.MODULE)
    retrievedValue should be (linear)
  }

  "NHWC DataFormat conversion " should " work properly" in {
    val format : DataFormat = NHWC
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, format, universe.typeOf[DataFormat])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    attr.getDataType should be (DataType.DATA_FORMAT)
    retrievedValue should be (format)
  }

  "NCHW DataFormat conversion " should " work properly" in {
    val format : DataFormat = NCHW
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, format, universe.typeOf[DataFormat])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr)
    attr.getDataType should be (DataType.DATA_FORMAT)
    retrievedValue should be (format)
  }

  "Array of int32 conversion " should " work properly " in {
    val arry = Array[Int](1, 2, 3)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Int]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of int64 conversion " should " work properly " in {
    val arry = Array[Long](1L, 2L, 3L)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Long]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of float conversion " should " work properly " in {
    val arry = Array[Float](1.0f, 2.0f, 3.0f)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Float]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Null Array of float conversion " should " work properly " in {
    val arry : Array[Float] = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Float]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of double conversion " should " work properly " in {
    val arry = Array[Double](1.0, 2.0, 3.0)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Double]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }


  "Array of String conversion " should " work properly" in {
    val arry = new Array[String](2)
    arry(0) = "test1"
    arry(1) = "test2"
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[String]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of Boolean conversion " should " work properly" in {
    val arry = Array[Boolean](true, false)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Boolean]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of Regularizer conversion " should " work properly" in {
    val arry = new Array[Regularizer[Float]](2)
    arry(0) = L2Regularizer(1.0)
    arry(1) = L1Regularizer(1.0)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[Regularizer[Float]]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of Tensor conversion" should "work properly" in {
    val tensor1 = Tensor(2, 3).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor(2, 3).apply1(_ => Random.nextFloat())
    val tensorArray = Array(tensor1, tensor2)
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, tensorArray, universe.typeOf[Array[Tensor[Float]]])

    attriBulder.clearArrayValue()
    val arrayValue = ArrayValue.newBuilder
    arrayValue.setDatatype(DataType.TENSOR)
    tensorArray.foreach(t => {
      val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
        identityHashCode(t)).get.asInstanceOf[BigDLTensor])
      tensorBuilder.setStorage(map.get(System.
        identityHashCode(t.storage.array())).get.asInstanceOf[TensorStorage])
      arrayValue.addTensor(tensorBuilder.build)
    })
    arrayValue.setSize(2)
    attriBulder.setArrayValue(arrayValue.build)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue.isInstanceOf[Array[Tensor[_]]] should be (true)
    retrievedValue should be (tensorArray)
  }

  "Array of VariableFormat conversion " should " work properly" in {
    val arry = new Array[VariableFormat](1)
    arry(0) = Default
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[VariableFormat]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of Init methods conversion " should " work properly" in {
    val arry = new Array[InitializationMethod](2)
    arry(0) = RandomUniform
    arry(1) = Zeros
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[InitializationMethod]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Array of Dataformat conversion " should " work properly" in {
    val arry = new Array[DataFormat](2)
    arry(0) = NCHW
    arry(1) = NHWC
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry, universe.typeOf[Array[DataFormat]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "Null Array of Modules conversion" should " work properly" in {
    val arry : Array[AbstractModule[Activity, Activity, Float]] = null
    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, null),
      attriBulder, arry,
      universe.typeOf[Array[AbstractModule[Activity, Activity, Float]]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, null), attr)
    retrievedValue should be (arry)
  }

  "NameList conversion" should "work properly" in {

    val map1 = new mutable.HashMap[String, mutable.Map[String, Any]]

    val attrsMap = new mutable.HashMap[String, Any]

    attrsMap("int") = 1
    attrsMap("long") = 2L
    attrsMap("float") = 3.0f
    attrsMap("double") = 4.0
    attrsMap("string") = "str"
    attrsMap("bool") = true
    attrsMap("dataformat") = NCHW

    map1("test") = attrsMap

    val attriBulder = AttrValue.newBuilder
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBulder, map1)
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attr).
      asInstanceOf[mutable.HashMap[String, mutable.Map[String, Any]]]

    retrievedValue should be (map1)

  }

  "QuantizedTensor" should "work properly" in {
    val bytes = new Array[Byte](5)
    val min = Array[Float]('H')
    val max = Array[Float]('O')
    val sum = Array[Float]("HELLO".sum)
    "HELLO".zipWithIndex.foreach(x => bytes(x._2) = x._1.toByte)
    bytes.foreach(x => println(x.toChar))
    val tensor = QuantizedTensor[Float](bytes, max, min, sum, Array(1, 5), LinearWeightParams(1, 5))

    map.clear()
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType), attriBulder,
      tensor, ModuleSerializer.tensorType)

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(tensor)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(tensor.getStorage)).get.asInstanceOf[TensorStorage])
    attriBulder.setTensorValue(tensorBuilder.build)

    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map,
      ProtoStorageType), attr)
    attr.getDataType should be (DataType.TENSOR)

    retrievedValue.hashCode() should be (tensor.hashCode())
  }

  "Array of QuantizedTensor" should "work properly" in {
    val bytes = new Array[Byte](5)
    val min = Array[Float]('H')
    val max = Array[Float]('O')
    val sum = Array[Float]("HELLO".sum)
    "HELLO".zipWithIndex.foreach(x => bytes(x._2) = x._1.toByte)
    bytes.foreach(x => println(x.toChar))
    val tensor1 = QuantizedTensor[Float](bytes, max, min, sum, Array(1, 5),
      LinearWeightParams(1, 5))
    val tensor2 = QuantizedTensor[Float](bytes, max, min, sum, Array(1, 5),
      LinearWeightParams(1, 5))
    val array = new Array[QuantizedTensor[Float]](2)
    array(0) = tensor1
    array(1) = tensor2

    map.clear()
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType), attriBulder,
      array, universe.typeOf[Array[QuantizedTensor[Float]]])

    attriBulder.clearArrayValue()
    val arrayValue = ArrayValue.newBuilder
    arrayValue.setDatatype(DataType.TENSOR)
    array.foreach(t => {
      val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
        identityHashCode(t)).get.asInstanceOf[BigDLTensor])
      tensorBuilder.setStorage(map.get(System.
        identityHashCode(t.getStorage)).get.asInstanceOf[TensorStorage])
      arrayValue.addTensor(tensorBuilder.build)
    })
    arrayValue.setSize(2)
    attriBulder.setArrayValue(arrayValue.build)

    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.getAttributeValue(DeserializeContext(null, map,
      ProtoStorageType), attr)
  }

  "Single Shape converter" should "work properly" in {
    val shape = SingleShape(List(1, 3, 4))
    map.clear()
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType), attriBulder,
      shape, universe.typeOf[BigDLShape])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType),
        attriBulder.build)

    shape should be (retrievedValue)

  }

  "Multiple shape converter" should "work properly" in {
    val shape1 = SingleShape(List(1, 3, 4))
    val shape2 = SingleShape(List(1, 3, 4))

    val mul1 = MultiShape(List(shape1, shape2))

    val shape3 = SingleShape(List(1, 3, 4))

    val mul2 = MultiShape(List(shape3, mul1))

    map.clear()
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType), attriBulder,
      mul2, universe.typeOf[BigDLShape])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType),
        attriBulder.build)

    mul2 should be (retrievedValue)
  }

  "Array of shape converter" should "work properly" in {
    val shape1 = SingleShape(List(1, 3, 4))
    val shape2 = SingleShape(List(1, 3, 4))
    val array = Array[BigDLShape](shape1, shape2)
    map.clear()
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType), attriBulder,
      array, universe.typeOf[Array[BigDLShape]])
    val attr = attriBulder.build
    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType),
        attriBulder.build)

    array should be (retrievedValue)

  }
}
