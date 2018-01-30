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

package com.intel.analytics.bigdl.utils.serializer.converters

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType, InitMethod, InitMethodType}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.InitializationMethod]]
 */
object InitMethodConverter extends DataConverter {

  override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
                                             (implicit ev: TensorNumeric[T]): AnyRef = {
    val initMemethod = attribute.getInitMethodValue
    val initType = initMemethod.getMethodType
    val methodData = initMemethod.getDataList
    initType match {
      case InitMethodType.RANDOM_UNIFORM => RandomUniform
      case InitMethodType.RANDOM_UNIFORM_PARAM =>
        RandomUniform(methodData.get(0), methodData.get(1))
      case InitMethodType.RANDOM_NORMAL =>
        RandomNormal(methodData.get(0), methodData.get(1))
      case InitMethodType.ZEROS => Zeros
      case InitMethodType.ONES => Ones
      case InitMethodType.CONST => ConstInitMethod(methodData.get(0))
      case InitMethodType.XAVIER => Xavier
      case InitMethodType.BILINEARFILLER => BilinearFiller
      case InitMethodType.EMPTY_INITIALIZATION => null
    }
  }

  override def setAttributeValue[T: ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
    value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
    attributeBuilder.setDataType(DataType.INITMETHOD)
    val initMethodBuilder = InitMethod.newBuilder
    if (value != null) {
      val initMethod = value.asInstanceOf[InitializationMethod]
      initMethod match {
        case RandomUniform =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM)
        case ru: RandomUniform =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM_PARAM)
          initMethodBuilder.addData(ru.lower)
          initMethodBuilder.addData(ru.upper)
        case rm: RandomNormal =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_NORMAL)
          initMethodBuilder.addData(rm.mean)
          initMethodBuilder.addData(rm.stdv)
        case Zeros =>
          initMethodBuilder.setMethodType(InitMethodType.ZEROS)
        case Ones =>
          initMethodBuilder.setMethodType(InitMethodType.ONES)
        case const: ConstInitMethod =>
          initMethodBuilder.setMethodType(InitMethodType.CONST)
          initMethodBuilder.addData(const.value)
        case Xavier =>
          initMethodBuilder.setMethodType(InitMethodType.XAVIER)
        case BilinearFiller =>
          initMethodBuilder.setMethodType(InitMethodType.BILINEARFILLER)
      }
      attributeBuilder.setInitMethodValue(initMethodBuilder.build)
    } else {
      initMethodBuilder.setMethodType(InitMethodType.EMPTY_INITIALIZATION)
      attributeBuilder.setInitMethodValue(initMethodBuilder.build)
    }
  }
}

