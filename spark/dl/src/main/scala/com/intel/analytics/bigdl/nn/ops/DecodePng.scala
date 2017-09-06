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

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class DecodePng[T: ClassTag](channels: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[ByteString], Tensor[Int], T] {
  override def updateOutput(input: Tensor[ByteString]): Tensor[Int] = {
    require(input.isScalar, "only support ByteString scalar")
    val image = ImageIO.read(new ByteArrayInputStream(input.valueAt().toByteArray))
    require(image != null, "Can't decode image")
    val imageWidth = image.getWidth
    val imageHeight = image.getHeight

    val expectedChannels = if (channels == 0) {
      image.getColorModel.getNumComponents
    } else {
      require(channels == image.getColorModel.getNumComponents,
        "Only support inputs channels equal to desired channels")
      channels
    }

    val data = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    require(data.length % channels == 0,
      s"input data length is not a multiple of channels: $channels")

    output.resize(imageHeight, imageWidth, expectedChannels)

    val outputData = output.storage().array()
    val offset = output.storageOffset() - 1

    bytesToInts(data, outputData, offset)
    output
  }

  private def bytesToInts(bytes: Array[Byte], ints: Array[Int], start: Int): Unit = {
    val length = bytes.length
    var i = 0
    while (i < length) {
      ints(i + start) = bytes(i) & 0xff
      i += 1
    }
  }

  override def updateGradInput(input: Tensor[ByteString],
                               gradOutput: Tensor[Int]): Tensor[ByteString] = {
    throw new UnsupportedOperationException("no backward on ParseExample")
  }
}
