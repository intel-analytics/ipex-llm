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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.tensor.{FloatType, LongType, StringType, Tensor}
import com.google.protobuf.{ByteString, CodedOutputStream}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}
import org.tensorflow.example._
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

class ParseSingleExampleSpec extends FlatSpec with Matchers {

  "ParseSingleExample" should "be able to parse a example" in {

    val floatBuilder = FloatList.newBuilder()
      .addValue(0.0f).addValue(1.0f).addValue(2.0f)
    val floatFeature = Feature.newBuilder().setFloatList(floatBuilder).build()

    val longBuilder = Int64List.newBuilder()
      .addValue(0).addValue(1).addValue(2)
    val longFeature = Feature.newBuilder().setInt64List(longBuilder).build()

    val bytesBuilder = BytesList.newBuilder().addValue(ByteString.copyFromUtf8("abcd"))
    val bytesFeature = Feature.newBuilder().setBytesList(bytesBuilder).build()

    val features = Features.newBuilder()
      .putFeature("floatFeature", floatFeature)
      .putFeature("longFeature", longFeature)
      .putFeature("bytesFeature", bytesFeature)
    val example = Example.newBuilder().setFeatures(features).build()
    val length = example.getSerializedSize
    val data = new Array[Byte](length)
    val outputStream = CodedOutputStream.newInstance(data)
    example.writeTo(outputStream)

    val key1 = ByteString.copyFromUtf8("floatFeature")
    val key2 = ByteString.copyFromUtf8("longFeature")
    val key3 = ByteString.copyFromUtf8("bytesFeature")
    val denseKeys = Seq(key1, key2, key3)

    val exampleParser = new ParseSingleExample[Float](
      Seq(FloatType, LongType, StringType), denseKeys, Seq(Array(3), Array(3), Array()))

    val serialized = Tensor[ByteString](Array(ByteString.copyFrom(data)), Array[Int](1))

    val input = T(serialized)

    val output = exampleParser.forward(input)

    val floatTensor = output(1).asInstanceOf[Tensor[Float]]
    val longTensor = output(2).asInstanceOf[Tensor[Long]]
    val stringTensor = output(3).asInstanceOf[Tensor[ByteString]]

    floatTensor should be (Tensor[Float](T(0.0f, 1.0f, 2.0f)))
    longTensor should be (Tensor[Long](T(0L, 1L, 2L)))
    stringTensor should be (Tensor.scalar((ByteString.copyFromUtf8("abcd"))))
  }

}

class ParseSingleExampleSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

    val floatBuilder = FloatList.newBuilder()
      .addValue(0.0f).addValue(1.0f).addValue(2.0f)
    val floatFeature = Feature.newBuilder().setFloatList(floatBuilder).build()

    val longBuilder = Int64List.newBuilder()
      .addValue(0).addValue(1).addValue(2)
    val longFeature = Feature.newBuilder().setInt64List(longBuilder).build()

    val bytesBuilder = BytesList.newBuilder().addValue(ByteString.copyFromUtf8("abcd"))
    val bytesFeature = Feature.newBuilder().setBytesList(bytesBuilder).build()

    val features = Features.newBuilder()
      .putFeature("floatFeature", floatFeature)
      .putFeature("longFeature", longFeature)
      .putFeature("bytesFeature", bytesFeature)
    val example = Example.newBuilder().setFeatures(features).build()
    val length = example.getSerializedSize
    val data = new Array[Byte](length)
    val outputStream = CodedOutputStream.newInstance(data)
    example.writeTo(outputStream)

    val key1 = ByteString.copyFromUtf8("floatFeature")
    val key2 = ByteString.copyFromUtf8("longFeature")
    val key3 = ByteString.copyFromUtf8("bytesFeature")
    val denseKeys = Seq(key1, key2, key3)

    val exampleParser = new ParseSingleExample[Float](Seq(FloatType, LongType, StringType),
      denseKeys, Seq(Array(3), Array(3), Array())).setName("parseSingleExample")

    val serialized = Tensor[ByteString](Array(ByteString.copyFrom(data)), Array[Int](1))

    val input = T(serialized)
    runSerializationTest(exampleParser, input)
  }
}
