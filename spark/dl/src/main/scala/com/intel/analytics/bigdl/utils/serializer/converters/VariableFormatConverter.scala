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

import com.intel.analytics.bigdl.nn.VariableFormat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType, VarFormat}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.VariableFormat]]
 */
object VariableFormatConverter extends DataConverter {

  override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
                                             (implicit ev: TensorNumeric[T]): AnyRef = {
    val format = attribute.getVariableFormatValue
    format match {
      case VarFormat.DEFAULT => VariableFormat.Default
      case VarFormat.ONE_D => VariableFormat.ONE_D
      case VarFormat.IN_OUT => VariableFormat.IN_OUT
      case VarFormat.OUT_IN => VariableFormat.OUT_IN
      case VarFormat.IN_OUT_KW_KH => VariableFormat.IN_OUT_KW_KH
      case VarFormat.OUT_IN_KW_KH => VariableFormat.OUT_IN_KW_KH
      case VarFormat.GP_OUT_IN_KW_KH => VariableFormat.GP_OUT_IN_KW_KH
      case VarFormat.GP_IN_OUT_KW_KH => VariableFormat.GP_IN_OUT_KW_KH
      case VarFormat.OUT_IN_KT_KH_KW => VariableFormat.OUT_IN_KT_KH_KW
      case VarFormat.EMPTY_FORMAT => null
    }
  }

  override def setAttributeValue[T: ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
    value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
    attributeBuilder.setDataType(DataType.VARIABLE_FORMAT)
    if (value != null) {
      val format = value.asInstanceOf[VariableFormat]
      val formatValue = format match {
        case VariableFormat.Default => VarFormat.DEFAULT
        case VariableFormat.ONE_D => VarFormat.ONE_D
        case VariableFormat.IN_OUT => VarFormat.IN_OUT
        case VariableFormat.OUT_IN => VarFormat.OUT_IN
        case VariableFormat.IN_OUT_KW_KH => VarFormat.IN_OUT_KW_KH
        case VariableFormat.OUT_IN_KW_KH => VarFormat.OUT_IN_KW_KH
        case VariableFormat.GP_OUT_IN_KW_KH => VarFormat.GP_OUT_IN_KW_KH
        case VariableFormat.GP_IN_OUT_KW_KH => VarFormat.GP_IN_OUT_KW_KH
        case VariableFormat.OUT_IN_KT_KH_KW => VarFormat.OUT_IN_KT_KH_KW
      }
      attributeBuilder.setVariableFormatValue(formatValue)
    } else {
      attributeBuilder.setVariableFormatValue(VarFormat.EMPTY_FORMAT)
    }
  }
}
