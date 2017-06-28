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

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn.InitializationMethod
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Model.{AttrValue, BigDLTensor, InitMethod}
import serialization.Model.AttrValue.DataType

import scala.reflect.ClassTag

trait DataConverter {
  def getAttributeValue[T : ClassTag](attribute: AttrValue)(
    implicit ev: TensorNumeric[T]) : AnyRef
  def setAttributeValue[T : ClassTag](attributeBuilder : AttrValue.Builder, value: AnyRef,
                                      valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]) : Unit
}

object DataConverter extends DataConverter{

  val regularizerCls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")

  def getAttributeValue[T : ClassTag](attribute: AttrValue)
    (implicit ev: TensorNumeric[T]) : AnyRef = {
    attribute.getDataType match {
      case DataType.INT32 => Integer.valueOf(attribute.getInt32Value)
      case DataType.INT64 => Long.box(attribute.getInt64Value)
      case DataType.DOUBLE => Double.box(attribute.getDoubleValue)
      case DataType.FLOAT => Float.box(attribute.getFloatValue)
      case DataType.BOOL => Boolean.box(attribute.getBoolValue)
      case DataType.REGULARIZER => RegularizerConverter.getAttributeValue(attribute)
      case DataType.TENSOR => TensorConverter.getAttributeValue(attribute)
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  def setAttributeValue[T : ClassTag](
    attributeBuilder : AttrValue.Builder, value: AnyRef, valueType : universe.Type)
    (implicit ev: TensorNumeric[T]): Unit = {
    if (valueType == universe.typeOf[Int]) {
      attributeBuilder.setDataType(DataType.INT32)
      attributeBuilder.setInt32Value(value.asInstanceOf[Int])
    } else if (valueType == universe.typeOf[Long]) {
      attributeBuilder.setDataType(DataType.INT64)
      attributeBuilder.setInt64Value(value.asInstanceOf[Long])
    } else if (valueType == universe.typeOf[Float]) {
      attributeBuilder.setDataType(DataType.FLOAT)
      attributeBuilder.setFloatValue(value.asInstanceOf[Float])
    } else if (valueType == universe.typeOf[Double]) {
      attributeBuilder.setDataType(DataType.DOUBLE)
      attributeBuilder.setDoubleValue(value.asInstanceOf[Double].toFloat)
    } else if (valueType == universe.typeOf[Boolean]) {
      attributeBuilder.setDataType(DataType.BOOL)
      attributeBuilder.setBoolValue(value.asInstanceOf[Boolean])
    } else if (valueType.toString == "com.intel.analytics.bigdl.optim.Regularizer[T]") {
      RegularizerConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == "com.intel.analytics.bigdl.tensor.Tensor[T]") {
      TensorConverter.setAttributeValue(attributeBuilder, value)
    }
  }

  object RegularizerConverter extends DataConverter {
    override def getAttributeValue[T : ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val regularizer = attribute.getRegularizerValue
      val regularizerType = regularizer.getRegularizerType
      if (regularizer.getRegularDataCount == 0) {
        return null
      }
      val l1 = regularizer.getRegularDataList.get(0)
      val l2 = regularizer.getRegularDataList.get(1)
      regularizerType match {
        case serialization.Model.RegularizerType.L1Regularizer => L1Regularizer[T](l1)
        case serialization.Model.RegularizerType.L2Regularizer => L2Regularizer[T](l2)
        case serialization.Model.RegularizerType.L1L2Regularizer => L1L2Regularizer[T](l1, l2)
      }
    }

    override def setAttributeValue[T : ClassTag]
    (attributeBuilder: AttrValue.Builder, value: AnyRef,
     valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.REGULARIZER)
      if (value != null) {
        var regularizerBuilder = serialization.Model.Regularizer.newBuilder
        val regularizer = value.asInstanceOf[L1L2Regularizer[T]]
        val l1 = regularizer.l1
        val l2 = regularizer.l2
        regularizerBuilder.addRegularData(l1)
        regularizerBuilder.addRegularData(l2)
        val regularizerType = regularizer match {
          case l1: L1Regularizer[_] => serialization.Model.RegularizerType.L1Regularizer
          case l2: L2Regularizer[_] => serialization.Model.RegularizerType.L2Regularizer
          case l1l2: L1L2Regularizer[_] => serialization.Model.RegularizerType.L1L2Regularizer
        }
        regularizerBuilder.setRegularizerType(regularizerType)
        attributeBuilder.setRegularizerValue(regularizerBuilder.build)
      }
    }
  }

  object TensorConverter extends DataConverter {
    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val serializedTensor = attribute.getTensorValue
      val data = serializedTensor.getDataList.asScala
      if (data.size == 0) {
        return null
      }
      val sizes = serializedTensor.getSizeList.asScala
      val strorageArray = new Array[T](data.size)
      var i = 0;
      while (i < data.size) {
        strorageArray(i) = ev.fromType[Double](data(i))
        i += 1
      }
      val sizeArray = new Array[Int](sizes.size)
      i = 0;
      while (i < sizes.size) {
        sizeArray(i) = sizes(i)
        i += 1
      }
      Tensor[T](strorageArray, sizeArray)
    }

    override def setAttributeValue[T: ClassTag]
      (attributeBuilder: AttrValue.Builder, value: AnyRef,
       valueType : universe.Type = null)
      (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.TENSOR)
      if (value != null) {
        val tensor = value.asInstanceOf[Tensor[T]]
        val tensorBuilder = BigDLTensor.newBuilder
        tensor.storage().array().foreach(data => tensorBuilder.addData(ev.toType[Double](data)))
        tensor.size().foreach(size => tensorBuilder.addSize(size))
        attributeBuilder.setTensorValue(tensorBuilder.build)
      }
    }
  }
}