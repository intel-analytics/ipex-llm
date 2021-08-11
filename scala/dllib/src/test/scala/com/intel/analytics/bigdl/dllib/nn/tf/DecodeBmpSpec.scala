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

import java.io.{File => JFile}

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import org.tensorflow.example.Example

class DecodeBmpSerialTest extends ModuleSerializationTest {
  private def getInputs(name: String): Tensor[ByteString] = {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    /* since the tfrecord file is loaded into byteArrays regardless of the
      original image type, we can map "bmp" to 0 as well
    */
    val index = name match {
      case "png" => 0
      case "jpeg" => 1
      case "gif" => 2
      case "raw" => 3
      case "bmp" => 0
    }

    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + JFile.separator + "decode_image_test_case.tfrecord"
    val file = new JFile(path)

    val bytesVector = TFRecordIterator(file).toVector
    val bmpBytes = bytesVector(index)

    val example = Example.parseFrom(bmpBytes)
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    Tensor[ByteString](Array(imageByteString), Array[Int]())
  }

  override def test(): Unit = {
    val decodeBmp = new DecodeBmp[Float](1).setName("decodeBmp")
    val input = getInputs("bmp")
    runSerializationTest(decodeBmp, input)
  }
}

