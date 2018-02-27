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

import java.io.File

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import org.tensorflow.example.Example
import org.tensorflow.framework.DataType

class DecodeRawSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val decodeRaw = new DecodeRaw[Float](DataType.DT_UINT8, true).setName("decodeRaw")
    val input = getInputs("raw")
    runSerializationTest(decodeRaw, input)
  }

  private def getInputs(name: String): Tensor[ByteString] = {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val index = name match {
      case "png" => 0
      case "jpeg" => 1
      case "gif" => 2
      case "raw" => 3
    }

    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + File.separator + "decode_image_test_case.tfrecord"
    val file = new File(path)

    val bytesVector = TFRecordIterator(file).toVector
    val pngBytes = bytesVector(index)

    val example = Example.parseFrom(pngBytes)
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    Tensor[ByteString](Array(imageByteString), Array[Int]())
  }
}
