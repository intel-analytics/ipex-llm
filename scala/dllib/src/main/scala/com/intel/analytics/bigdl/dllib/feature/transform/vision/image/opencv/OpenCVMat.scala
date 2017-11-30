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

/**
 * OpenCVMat is a Serializable wrapper of original Mat
 */
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

  /**
   * read local image path as opencv mat
   *
   * @param path local image path
   * @return mat
   */
  def read(path: String): OpenCVMat = {
    val bytes = FileUtils.readFileToByteArray(new File(path))
    fromImageBytes(bytes)
  }

  /**
   * convert image file in bytes to opencv mat
   *
   * @param fileContent bytes from an image file
   * @return opencv mat
   */
  def fromImageBytes(fileContent: Array[Byte]): OpenCVMat = {
    var mat: Mat = null
    var matOfByte: MatOfByte = null
    var result: OpenCVMat = null
    try {
      matOfByte = new MatOfByte(fileContent: _*)
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

  /**
   * convert opencv mat to image bytes
   *
   * @param mat opencv mat
   * @param encoding encoding type
   * @return bytes that represent an image
   */
  def imencode(mat: OpenCVMat, encoding: String = "png"): Array[Byte] = {
    val buf = new MatOfByte()
    try {
      Imgcodecs.imencode("." + encoding, mat, buf)
      buf.toArray
    } finally {
      buf.release()
    }
  }

  /**
   * Convert float array(pixels) to OpenCV mat
   *
   * @param floats float array that represents pixels
   * @param height image height
   * @param width image width
   * @return image in mat
   */
  def fromFloats(floats: Array[Float], height: Int, width: Int): OpenCVMat = {
    val mat = new Mat(height, width, CvType.CV_32FC3)
    mat.put(0, 0, floats)
    new OpenCVMat(mat)
  }

  /**
   * convert mat to byte array that represents pixels
   *
   * @param input opencv mat
   * @param buffer
   * @return
   */
  def toBytePixels(input: Mat, buffer: Array[Byte] = null): (Array[Byte], Int, Int) = {
    var bytes = buffer
    val length = input.channels() * input.height() * input.width()
    if (null == buffer || buffer.length < length) {
      bytes = new Array[Byte](length)
    }
    input.get(0, 0, bytes)
    (bytes, input.height(), input.width())
  }


  /**
   * convert mat to float array that represents pixels
   *
   * @param input mat
   * @param buffer float array
   * @return
   */
  def toFloatPixels(input: Mat,
    buffer: Array[Float] = null): (Array[Float], Int, Int) = {
    var floats = buffer
    val length = input.channels() * input.height() * input.width()
    if (null == buffer || buffer.length < length) {
      floats = new Array[Float](length)
    }
    if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(input, CvType.CV_32FC3)
    }
    input.get(0, 0, floats)
    (floats, input.height(), input.width())
  }
}
