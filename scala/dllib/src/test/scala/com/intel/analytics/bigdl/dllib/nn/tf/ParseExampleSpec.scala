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

class ParseExampleSpec extends FlatSpec with Matchers {

  "ParseExample" should "be able to parse a example" in {

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

    val exampleParser = new ParseExample[Float](3,
      Seq(FloatType, LongType, StringType), Seq(Array(3), Array(3), Array()))

    val serialized = Tensor[ByteString](Array(ByteString.copyFrom(data)), Array[Int](1))
    val names = Tensor[ByteString]()
    val key1 = Tensor[ByteString](Array(ByteString.copyFromUtf8("floatFeature")), Array[Int]())
    val key2 = Tensor[ByteString](Array(ByteString.copyFromUtf8("longFeature")), Array[Int]())
    val key3 = Tensor[ByteString](Array(ByteString.copyFromUtf8("bytesFeature")), Array[Int]())

    val default1 = Tensor[Float]()
    val default2 = Tensor[Long]()
    val default3 = Tensor[ByteString]()

    val input = T(serialized, names, key1, key2, key3, default1, default2, default3)

    val output = exampleParser.forward(input)

    val floatTensor = output(1).asInstanceOf[Tensor[Float]]
    val longTensor = output(2).asInstanceOf[Tensor[Long]]
    val stringTensor = output(3).asInstanceOf[Tensor[ByteString]]

    floatTensor should be (Tensor[Float](T(T(0.0f, 1.0f, 2.0f))))
    longTensor should be (Tensor[Long](T(T(0L, 1L, 2L))))
    stringTensor should be (Tensor[ByteString](
      Array(ByteString.copyFromUtf8("abcd")), Array[Int](1)))
  }

}

class ParseExampleSerialTest extends ModuleSerializationTest {
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

    val exampleParser = new ParseExample[Float](3, Seq(FloatType, LongType, StringType),
      Seq(Array(3), Array(3), Array())).setName("parseExample")

    val serialized = Tensor[ByteString](Array(ByteString.copyFrom(data)), Array[Int](1))
    val names = Tensor[ByteString]()
    val key1 = Tensor[ByteString](Array(ByteString.copyFromUtf8("floatFeature")), Array[Int]())
    val key2 = Tensor[ByteString](Array(ByteString.copyFromUtf8("longFeature")), Array[Int]())
    val key3 = Tensor[ByteString](Array(ByteString.copyFromUtf8("bytesFeature")), Array[Int]())

    val default1 = Tensor[Float]()
    val default2 = Tensor[Long]()
    val default3 = Tensor[ByteString]()
    val input = T(serialized, names, key1, key2, key3, default1, default2, default3)
    runSerializationTest(exampleParser, input)
  }
}
