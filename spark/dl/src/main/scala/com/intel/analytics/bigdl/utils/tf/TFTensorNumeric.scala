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
package com.intel.analytics.bigdl.utils.tf

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.{ConvertableFrom, StringType, TensorDataType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.UndefinedTensorNumeric

import scala.language.implicitConversions

object TFTensorNumeric {

  implicit object NumericByteString extends UndefinedTensorNumeric[ByteString]("ByteString") {

    override def getType(): TensorDataType = StringType
    override def plus(x: ByteString, y: ByteString): ByteString = x.concat(y)


    override def fromType[K](k: K)(implicit c: ConvertableFrom[K]): ByteString = {
      ByteString.copyFromUtf8(k.toString)
    }

    override def axpy(n: Int, da: ByteString, dx: Array[ByteString],
                      _dx_offset: Int, incx: Int, dy: Array[ByteString],
                      _dy_offset: Int, incy: Int): Unit = {
      var i = 0
      while (i < n) {
        dy(i + _dy_offset) = dx(_dx_offset + i).concat(dy(_dy_offset + i))
        i += 1
      }
    }

    override def nearlyEqual(a: ByteString, b: ByteString, epsilon: Double): Boolean = {
      a == b
    }

  }
}


