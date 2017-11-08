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

import java.io.File
import java.nio.ByteOrder

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.tensorflow.framework.TensorProto

import scala.collection.JavaConverters._

class TFUtilsSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private var constTensors: Map[String, TensorProto] = null
  before {
    constTensors = getConstTensorProto()
  }

  private def getConstTensorProto(): Map[String, TensorProto] = {
    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + File.separator + "consts.pbtxt"
    val nodes = TensorflowLoader.parseTxt(path)
    nodes.asScala.map(node => node.getName -> node.getAttrMap.get("value").getTensor).toMap
  }

  "parseTensor " should "work with bool TensorProto" in {
    val tensorProto = constTensors("bool_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Boolean](T(true, false, true, false)))
  }

  "parseTensor " should "work with float TensorProto" in {
    val tensorProto = constTensors("float_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Float](T(1.0f, 2.0f, 3.0f, 4.0f)))
  }

  "parseTensor " should "work with double TensorProto" in {
    val tensorProto = constTensors("double_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Double](T(1.0, 2.0, 3.0, 4.0)))
  }

  "parseTensor " should "work with int TensorProto" in {
    val tensorProto = constTensors("int_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Int](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with long TensorProto" in {
    val tensorProto = constTensors("long_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Long](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with int8 TensorProto" in {
    val tensorProto = constTensors("int8_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Int](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with uint8 TensorProto" in {
    val tensorProto = constTensors("uint8_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Int](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with int16 TensorProto" in {
    val tensorProto = constTensors("int16_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Int](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with uint16 TensorProto" in {
    val tensorProto = constTensors("uint16_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    bigdlTensor should be (Tensor[Int](T(1, 2, 3, 4)))
  }

  "parseTensor " should "work with string TensorProto" in {
    import TFTensorNumeric.NumericByteString
    val tensorProto = constTensors("string_const")
    val bigdlTensor = TFUtils.parseTensor(tensorProto, ByteOrder.LITTLE_ENDIAN)
    val data = Array(
      ByteString.copyFromUtf8("a"),
      ByteString.copyFromUtf8("b"),
      ByteString.copyFromUtf8("c"),
      ByteString.copyFromUtf8("d")
    )
    bigdlTensor should be (Tensor[ByteString](data, Array[Int](4)))
  }
}
