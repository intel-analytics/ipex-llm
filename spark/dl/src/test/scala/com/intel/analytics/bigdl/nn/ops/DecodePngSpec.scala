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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import org.scalatest.{FlatSpec, Matchers}
import java.io.{File => JFile}

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import org.tensorflow.example.Example

class DecodePngSpec extends FlatSpec with Matchers {

  "DecodePng " should "be able to decode png" in {
    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + JFile.separator + "mnist_train.tfrecord"
    val file = new JFile(path)

    val iter = new TFRecordIterator(file)
    val example = Example.parseFrom(iter.next())
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    val input = Tensor[ByteString](imageByteString)

    val decoder = new DecodePng[Int](1)

    val output = decoder.forward(input)

    output.size() should be (Array(28, 28, 1))
  }

}
