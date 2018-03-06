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

import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import org.scalatest.{FlatSpec, Matchers}
import java.io.{File => JFile}
import java.nio.{ByteBuffer, ByteOrder}

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.tensorflow.example.Example
import org.tensorflow.framework.DataType
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

class DecodeImageSpec extends FlatSpec with Matchers {

  "DecodeRaw " should "be able to decode raw bytes" in {

    val input = getInputs("raw")

    val decoder = new DecodeRaw[Float](DataType.DT_UINT8, true)

    val output = decoder.forward(input).asInstanceOf[Tensor[Int]]

    output.size() should be (Array(28*28))
  }

  "DecodeRaw " should "be able to decode float raw bytes" in {

    val data = ByteBuffer.allocate(16)
    data.order(ByteOrder.LITTLE_ENDIAN)
    data.putFloat(1.0f)
    data.putFloat(2.0f)
    data.putFloat(3.0f)
    data.putFloat(4.0f)

    val input = Tensor.scalar(ByteString.copyFrom(data.array()))

    val decoder = new DecodeRaw[Float](DataType.DT_FLOAT, true)

    val output = decoder.forward(input).asInstanceOf[Tensor[Int]]

    output should be (Tensor[Float](Array(1.0f, 2.0f, 3.0f, 4.0f), Array(4)))
  }

  "DecodePng " should "be able to decode png" in {

    val input = getInputs("png")

    val decoder = new DecodePng[Int](1)

    val output = decoder.forward(input)
    val expected = getRaw()

    output should be (expected)
  }

  "DecodeJpeg " should "be able to decode jpeg" in {
    val input = getInputs("jpeg")

    val decoder = new DecodeJpeg[Int](1)

    val output = decoder.forward(input)

    output.size() should be (Array(28, 28, 1))
  }

  "DecodeGif " should "be able to decode gif" in {
    val input = getInputs("gif")

    val decoder = new DecodeGif[Int]()

    val output = decoder.forward(input)

    output.size() should be (Array(1, 28, 28, 3))

  }

  private def getRaw(): Tensor[Int] = {
    val input = getInputs("raw")

    val decoder = new DecodeRaw[Float](DataType.DT_UINT8, true)

    val output = decoder.forward(input).asInstanceOf[Tensor[Int]]

    output.resize(Array(28, 28, 1))
  }

  private def getInputs(name: String): Tensor[ByteString] = {
    val index = name match {
      case "png" => 0
      case "jpeg" => 1
      case "gif" => 2
      case "raw" => 3
    }

    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + JFile.separator + "decode_image_test_case.tfrecord"
    val file = new JFile(path)

    val bytesVector = TFRecordIterator(file).toVector
    val pngBytes = bytesVector(index)

    val example = Example.parseFrom(pngBytes)
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    Tensor[ByteString](Array(imageByteString), Array[Int]())
  }

}

class DecodeImageSerialTest extends ModuleSerializationTest {
  private def getInputs(name: String): Tensor[ByteString] = {
    val index = name match {
      case "png" => 0
      case "jpeg" => 1
      case "gif" => 2
      case "raw" => 3
    }

    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + JFile.separator + "decode_image_test_case.tfrecord"
    val file = new JFile(path)

    val bytesVector = TFRecordIterator(file).toVector
    val pngBytes = bytesVector(index)

    val example = Example.parseFrom(pngBytes)
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    Tensor[ByteString](Array(imageByteString), Array[Int]())
  }

  override def test(): Unit = {
    val decodeImage = new DecodeImage[Float](1).setName("decodeImage")
    val input = getInputs("png")
    runSerializationTest(decodeImage, input)
  }
}
