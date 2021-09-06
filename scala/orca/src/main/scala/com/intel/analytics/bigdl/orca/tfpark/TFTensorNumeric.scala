/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.tensor.TensorNumericMath.UndefinedTensorNumeric
import com.intel.analytics.bigdl.tensor.{ConvertableFrom, StringType, TensorDataType}

import scala.language.implicitConversions

object TFTensorNumeric {

  implicit object NumericByteArray extends UndefinedTensorNumeric[Array[Byte]]("ByteArray") {

    override def getType(): TensorDataType = StringType

    override def fromType[K](k: K)(implicit c: ConvertableFrom[K]): Array[Byte] = {
      k.toString.getBytes("UTF-8")
    }

  }
}
