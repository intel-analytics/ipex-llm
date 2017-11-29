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

package com.intel.analytics.bigdl.transform.vision.image.opencv

import java.io.{File, IOException, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.opencv.OpenCV
import org.apache.commons.io.FileUtils
import org.opencv.core.{CvType, Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs

class OpenCVMat() extends Mat with Serializable {

  def this(mat: Mat) {
    this()
    mat.copyTo(this)
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.writeInt(rows())
    out.writeInt(cols())
    out.writeInt(`type`())
    val size = (elemSize() * rows() * cols()).toInt
    out.writeInt(size)
    val bytes = new Array[Byte](size)
    get(rows(), cols(), bytes)
    out.write(bytes)
  }

  @throws(classOf[IOException])
  private def readObject(input: ObjectInputStream): Unit = {
    val rows = input.readInt()
    val cols = input.readInt()
    val t = input.readInt()
    val size = input.readInt()
    val data = new Array[Byte](size)
    input.read(data)
    create(rows, cols, t)
    put(rows, cols, data)
  }

  var isReleased: Boolean = false
  override def release(): Unit = {
    super.release()
    isReleased = true
  }

  /**
   * get shape of mat
   * @return (height, width, channel)
   */
  def shape(): (Int, Int, Int) = {
    (height(), width(), channels())
  }
}

object OpenCVMat {
  OpenCV.isOpenCVLoaded

  def read(path: String): OpenCVMat = {
    val bytes = FileUtils.readFileToByteArray(new File(path))
    toMat(bytes)
  }

  def toMat(bytes: Array[Byte]): OpenCVMat = {
    var mat: Mat = null
    var matOfByte: MatOfByte = null
    var result: OpenCVMat = null
    try {
      matOfByte = new MatOfByte(bytes: _*)
      mat = Imgcodecs.imdecode(matOfByte, Imgcodecs.CV_LOAD_IMAGE_COLOR)
      result = new OpenCVMat(mat)
    } catch {
      case e: Exception =>
        if (null != result) result.release()
        throw e
    } finally {
      if (null != mat) mat.release()
      if (null != matOfByte) matOfByte.release()
    }
    result
  }

  def toBytes(mat: Mat, encoding: String = "png"): Array[Byte] = {
    val buf = new MatOfByte()
    try {
      Imgcodecs.imencode("." + encoding, mat, buf)
      buf.toArray
    } finally {
      buf.release()
    }
  }

  def toBytesBuf(mat: Mat, bytes: Array[Byte]): Array[Byte] = {
    require(bytes.length == mat.channels() * mat.height() * mat.width())
    mat.get(0, 0, bytes)
    bytes
  }

  def toFloatBuf(input: Mat, floats: Array[Float], buf: Mat = null): Array[Float] = {
    val bufMat = if (buf == null) new OpenCVMat() else buf
    val floatMat = if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(bufMat, CvType.CV_32FC3)
      bufMat
    } else {
      input
    }
    require(floats.length >= input.channels() * input.height() * input.width())
    floatMat.get(0, 0, floats)
    if (null == buf) {
      bufMat.release()
    }
    floats
  }

  def toFloats(input: Mat, floats: Array[Float], buf: Array[Byte] = null): Array[Float] = {
    val width = input.width()
    val height = input.height()
    val bytes = if (null != buf) buf else new Array[Byte](width * height * 3)
    val rawData = toBytesBuf(input, bytes)
    var i = 0
    while (i < width * height * 3) {
      floats(i) = rawData(i) & 0xff
      i += 1
    }
    floats
  }

  def floatToMat(floats: Array[Float], height: Int, width: Int): OpenCVMat = {
    val mat = new Mat(height, width, CvType.CV_32FC3)
    mat.put(0, 0, floats)
    new OpenCVMat(mat)
  }
}
