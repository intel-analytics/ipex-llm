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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLTensor, TensorStorage}

import scala.collection.mutable
import scala.reflect.runtime.universe


class TensorConversionSpec extends FlatSpec with Matchers{

  val map = new mutable.HashMap[Int, Any]()

  "ByteString tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

    val byteString = Tensor[ByteString](Array(ByteString.copyFromUtf8("a"),
      ByteString.copyFromUtf8("b")), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, byteString, universe.typeOf[Tensor[ByteString]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(byteString)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(byteString.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    byteString should be (retrievedValue)

  }

  "Char tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericChar

    val chars = Tensor[Char](Array('a', 'b'), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, chars, universe.typeOf[Tensor[Char]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(chars)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(chars.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    chars should be (retrievedValue)

  }


  "Int tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericInt

    val ints = Tensor[Int](Array(2, 3), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, ints, universe.typeOf[Tensor[Int]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(ints)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(ints.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    ints should be (retrievedValue)

  }

  "Long tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericLong

    val longs = Tensor[Long](Array(2L, 3L), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, longs, universe.typeOf[Tensor[Long]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(longs)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(longs.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    longs should be (retrievedValue)

  }

  "Short tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericShort

    val shorts = Tensor[Short](Array[Short](2, 3), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, shorts, universe.typeOf[Tensor[Short]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(shorts)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(shorts.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    shorts should be (retrievedValue)

  }

  "Float tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val floats = Tensor[Float](Array[Float](2f, 3f), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, floats, universe.typeOf[Tensor[Float]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(floats)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(floats.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    floats should be (retrievedValue)

  }

  "Double tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble

    val doubles = Tensor[Double](Array[Double](2, 3), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, doubles, universe.typeOf[Tensor[Double]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(doubles)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(doubles.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    doubles should be (retrievedValue)

  }

  "String tensor conversion " should "work properly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericString

    val strings = Tensor[String](Array[String]("hello", "world"), Array(2))

    val attriBuilder = AttrValue.newBuilder()
    map.clear()
    DataConverter.setAttributeValue(SerializeContext(null, map, ProtoStorageType),
      attriBuilder, strings, universe.typeOf[Tensor[String]])

    val tensorBuilder = BigDLTensor.newBuilder(map.get(System.
      identityHashCode(strings)).get.asInstanceOf[BigDLTensor])
    tensorBuilder.setStorage(map.get(System.
      identityHashCode(strings.storage.array())).get.asInstanceOf[TensorStorage])

    attriBuilder.setTensorValue(tensorBuilder.build)

    map.clear()
    val retrievedValue = DataConverter.
      getAttributeValue(DeserializeContext(null, map, ProtoStorageType), attriBuilder.build)

    strings should be (retrievedValue)

  }

}
