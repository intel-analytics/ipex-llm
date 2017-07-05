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

import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.nn.VariableFormat.{Default, ONE_D}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import serialization.Model.AttrValue

import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import serialization.Model.AttrValue.DataType

import scala.util.Random


class DataConverterSpec extends FlatSpec with Matchers{

  "Primitive Int type conversion " should " work properly" in {
    val intValue = 1
    val attriBulder = AttrValue.newBuilder
    val intType = universe.typeOf[Int]
    DataConverter.setAttributeValue(attriBulder, intValue, intType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue should be (intValue)
  }

  "Primitive Long type conversion " should " work properly" in {
    val longValue = 1L
    val attriBulder = AttrValue.newBuilder
    val longType = universe.typeOf[Long]
    DataConverter.setAttributeValue(attriBulder, longValue, longType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue should be (longValue)
  }

  "Primitive Float type conversion " should " work properly" in {
    val floatValue = 1.0f
    val attriBulder = AttrValue.newBuilder
    val floatType = universe.typeOf[Float]
    DataConverter.setAttributeValue(attriBulder, floatValue, floatType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue should be (floatValue)
  }

  "Primitive Double type conversion " should " work properly" in {
    val doubleValue = 1.0
    val attriBulder = AttrValue.newBuilder
    val doubleType = universe.typeOf[Double]
    DataConverter.setAttributeValue(attriBulder, doubleValue, doubleType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue should be (doubleValue)
  }

  "Primitive String type conversion " should " work properly" in {
    val strValue = "test"
    val attriBulder = AttrValue.newBuilder
    val strType = universe.typeOf[String]
    DataConverter.setAttributeValue(attriBulder, strValue, strType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue should be (strValue)
  }

  "Primitive Boolean type conversion " should " work properly" in {
    val boolValue = false
    val attriBulder = AttrValue.newBuilder
    val boolType = universe.typeOf[Boolean]
    DataConverter.setAttributeValue(attriBulder, boolValue, boolType)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue.isInstanceOf[Boolean] should be (true)
    retrievedValue.asInstanceOf[Boolean] should be (boolValue)
  }

  "L1L2Regularizer conversion " should  " work properly" in {
    val regularizer = L1L2Regularizer(1.0, 2.0)
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")
    val tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).selfType
    DataConverter.setAttributeValue(attriBulder, regularizer, tpe)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue.isInstanceOf[L1L2Regularizer[Float]] should be (true)
    retrievedValue.asInstanceOf[L1L2Regularizer[Float]].l1 should be (regularizer.l1)
    retrievedValue.asInstanceOf[L1L2Regularizer[Float]].l2 should be (regularizer.l2)
  }

  "L1Regularizer conversion " should  " work properly" in {
    val regularizer = L1Regularizer(1.0)
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")
    val tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).selfType
    DataConverter.setAttributeValue(attriBulder, regularizer, tpe)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue.isInstanceOf[L1Regularizer[Float]] should be (true)
    retrievedValue.asInstanceOf[L1Regularizer[Float]].l1 should be (regularizer.l1)
  }

  "L2Regularizer conversion " should  " work properly" in {
    val regularizer = L2Regularizer(1.0)
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")
    val tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).selfType
    DataConverter.setAttributeValue(attriBulder, regularizer, tpe)
    val retrievedValue = DataConverter.getAttributeValue(attriBulder.build)
    retrievedValue.isInstanceOf[L2Regularizer[Float]] should be (true)
    retrievedValue.asInstanceOf[L2Regularizer[Float]].l2 should be (regularizer.l2)
  }

  "Empty Regularizer conversion " should  " work properly" in {
    val regularizer : L1L2Regularizer[Float] = null
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")
    val tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).toType
    DataConverter.setAttributeValue(attriBulder, regularizer, tpe)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.REGULARIZER)
    retrievedValue should be (regularizer)
  }
  "Tensor conversion " should " work properly" in {
    val tensor = Tensor(5, 5).apply1(e => Random.nextFloat())
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.tensor.Tensor")
    val tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).selfType
    DataConverter.setAttributeValue(attriBulder, tensor, tpe)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.TENSOR)
    retrievedValue should be (tensor)
  }
  "Empty Tensor conversion " should " work properly" in {
    val tensor : Tensor[Float] = null
    val attriBulder = AttrValue.newBuilder
    val cls = Class.forName("com.intel.analytics.bigdl.tensor.Tensor")
    var tpe = universe.runtimeMirror(cls.getClassLoader).classSymbol(cls).selfType
    DataConverter.setAttributeValue(attriBulder, tensor, tpe)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.TENSOR)
    retrievedValue should be (tensor)
  }

  "VariableFormat conversion " should " work properly" in {
    val format : VariableFormat = Default
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }

  "VariableFormat conversion With Param " should " work properly" in {
    val format : VariableFormat = ONE_D
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }


  "Empty VariableFormat conversion " should " work properly" in {
    val format : VariableFormat = null
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, format, universe.typeOf[VariableFormat])
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.VARIABLE_FORMAT)
    retrievedValue should be (format)
  }

  "Init Method conversion " should " work properly" in {
    val initMethod = RandomUniform
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, initMethod, universe.typeOf[InitializationMethod])
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.INITMETHOD)
    retrievedValue should be (initMethod)
  }

  "Empty Init Method conversion " should " work properly" in {
    val initMethod = RandomUniform
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, initMethod, universe.typeOf[InitializationMethod])
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    attr.getDataType should be (DataType.INITMETHOD)
    retrievedValue should be (initMethod)
  }

  "Module Conversion " should " work properly" in {
  }

  "Array of int32 conversion " should " work properly " in {
    val arry = Array[Int](1, 2, 3)
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, arry)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    retrievedValue should be (arry)
  }

  "Array of int64 conversion " should " work properly " in {
    val arry = Array[Long](1L, 2L, 3L)
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, arry)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    retrievedValue should be (arry)
  }

  "Array of Tensor conversion " should " work properly" in {
    val tensor1 = Tensor(2, 3).apply1(_ => Random.nextFloat())
    val tensor2 = Tensor(2, 3).apply1(_ => Random.nextFloat())
    val tensorArray = Array(tensor1, tensor2)
    val attriBulder = AttrValue.newBuilder
    DataConverter.setAttributeValue(attriBulder, tensorArray)
    val attr = attriBulder.build
    val retrievedValue = DataConverter.getAttributeValue(attr)
    retrievedValue.isInstanceOf[Array[Tensor[Float]]] should be (true)
  }
}
