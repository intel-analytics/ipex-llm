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

import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType, RegularizerType, Regularizer => SerializeRegularizer}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe


/**
 * DataConverter for [[com.intel.analytics.bigdl.optim.Regularizer]]
 */
object RegularizerConverter extends DataConverter {

  override def getAttributeValue[T : ClassTag](context: DeserializeContext,
                                               attribute: AttrValue)
                                              (implicit ev: TensorNumeric[T]): AnyRef = {
    val regularizer = attribute.getRegularizerValue
    val regularizerType = regularizer.getRegularizerType
    if (regularizer.getRegularDataCount == 0) {
      return null
    }
    regularizerType match {
      case RegularizerType.L1Regularizer =>
        val l1 = regularizer.getRegularDataList.get(0)
        L1Regularizer[T](l1)
      case RegularizerType.L2Regularizer =>
        val l2 = regularizer.getRegularDataList.get(1)
        L2Regularizer[T](l2)
      case RegularizerType.L1L2Regularizer =>
        val l1 = regularizer.getRegularDataList.get(0)
        val l2 = regularizer.getRegularDataList.get(1)
        L1L2Regularizer[T](l1, l2)
    }
  }

  override def setAttributeValue[T : ClassTag]
  (context: SerializeContext[T], attributeBuilder: AttrValue.Builder, value: Any,
   valueType : universe.Type = null)
  (implicit ev: TensorNumeric[T]): Unit = {
    attributeBuilder.setDataType(DataType.REGULARIZER)
    if (value != null) {
      var regularizerBuilder = SerializeRegularizer.newBuilder
      val regularizer = value.asInstanceOf[L1L2Regularizer[T]]
      val l1 = regularizer.l1
      val l2 = regularizer.l2
      regularizerBuilder.addRegularData(l1)
      regularizerBuilder.addRegularData(l2)
      val regularizerType = regularizer match {
        case l1: L1Regularizer[_] => RegularizerType.L1Regularizer
        case l2: L2Regularizer[_] => RegularizerType.L2Regularizer
        case l1l2: L1L2Regularizer[_] => RegularizerType.L1L2Regularizer
      }
      regularizerBuilder.setRegularizerType(regularizerType)
      attributeBuilder.setRegularizerValue(regularizerBuilder.build)
    }
  }

}
