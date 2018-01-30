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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import org.apache.commons.io.FileUtils
import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * OpenCVMat is a Serializable wrapper of org.opencv.core.Mat
 */
class OpenCVMat() extends Mat with Serializable {

  def this(mat: Mat) {
    this()
    mat.copyTo(this)
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    try {
      val bytes = OpenCVMat.imencode(this)
      out.writeInt(`type`())
      out.writeObject(bytes)
    } catch {
      case e: Exception =>
        out.writeInt(`type`())
        out.writeObject(Array[Byte]())
    }
  }

  @throws(classOf[IOException])
  private def readObject(input: ObjectInputStream): Unit = {
    val t = input.readInt()
    val data = input.readObject.asInstanceOf[Array[Byte]]
    if (data.length == 0) {
      create(0, 0, t)
    } else {
      val mat = OpenCVMat.fromImageBytes(data)
      mat.convertTo(this, t)
    }
  }

  var isReleased: Boolean = false

  override def release(): Unit = {
    super.release()
    isReleased = true
  }

  /**
   * get shape of mat
   *
   * @return (height, width, channel)
   */
  def shape(): (Int, Int, Int) = {
    (height(), width(), channels())
  }

  /**
   * draw bounding box on current mat
   * @param bbox bounding box
   * @param text text
   * @param font text fond
   * @param boxColor bounding box color
   * @param textColor text color
   * @return
   */
  def drawBoundingBox(bbox: BoundingBox, text: String,
    font: Int = Core.FONT_HERSHEY_COMPLEX_SMALL,
    boxColor: (Double, Double, Double) = (0, 255, 0),
    textColor: (Double, Double, Double) = (255, 255, 255)): this.type = {
    Imgproc.rectangle(this,
      new Point(bbox.x1, bbox.y1),
      new Point(bbox.x2, bbox.y2),
      new Scalar(boxColor._1, boxColor._2, boxColor._3), 3)
    Imgproc.putText(this, text,
      new Point(bbox.x1, bbox.y1 - 2),
      font, 1,
      new Scalar(textColor._1, textColor._2, textColor._3), 1)
    this
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
      mat = Imgcodecs.imdecode(matOfByte, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
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
  def fromFloats(floats: Array[Float], height: Int, width: Int, channel: Int = 3): OpenCVMat = {
    require(channel >= 1 && channel <= 4, s"channel $channel is out of range [1,4]")
    require(floats.length >= height * width * channel,
      s"pixels array length ${floats.length} is less than " +
        s"height*width*channel ${height * width * channel}")
    val mat = new OpenCVMat()
    mat.create(height, width, CvType.CV_32FC(channel))
    mat.put(0, 0, floats)
    mat
  }

  /**
   * convert mat to byte array that represents pixels
   *
   * @param input opencv mat
   * @param buffer
   * @return
   */
  def toBytePixels(input: Mat, buffer: Array[Byte] = null): (Array[Byte], Int, Int, Int) = {
    val channel = input.channels()
    // the mat need to be type CV_8UCX in order to get pixels byte array
    if (input.`type`() != CvType.CV_8UC(channel)) {
      input.convertTo(input, CvType.CV_8UC(channel))
    }
    var bytes = buffer
    val length = input.channels() * input.height() * input.width()
    if (null == buffer || buffer.length < length) {
      bytes = new Array[Byte](length)
    }
    input.get(0, 0, bytes)
    (bytes, input.height(), input.width(), channel)
  }


  /**
   * convert mat to float array that represents pixels
   *
   * @param input mat
   * @param buffer float array
   * @return
   */
  def toFloatPixels(input: Mat,
    buffer: Array[Float] = null): (Array[Float], Int, Int, Int) = {
    var floats = buffer
    val length = input.channels() * input.height() * input.width()
    if (null == buffer || buffer.length < length) {
      floats = new Array[Float](length)
    }
    val channel = input.channels()
    if (input.`type`() != CvType.CV_32FC(channel)) {
      input.convertTo(input, CvType.CV_32FC(channel))
    }
    input.get(0, 0, floats)
    (floats, input.height(), input.width(), channel)
  }

  /**
   * convert pixel bytes to OpenCVMat
   * @param pixels pixels in byte array
   * @param height image height
   * @param width image width
   */
  def fromPixelsBytes(pixels: Array[Byte], height: Int, width: Int, channel: Int = 3): OpenCVMat = {
    require(channel >= 1 && channel <= 4, s"channel $channel is out of range [1,4]")
    require(pixels.length >= height * width * channel,
      s"pixels array length ${pixels.length} is less than " +
        s"height*width*channel ${height * width * channel}")
    val mat = new OpenCVMat()
    mat.create(height, width, CvType.CV_8UC(channel))
    mat.put(0, 0, pixels)
    mat
  }

  /**
   * convert float tensor to OpenCVMat,
   * Note that if you want to convert the tensor to BGR image,
   * the element should be in range [0, 255]
   * @param tensor tensor that represent an image
   * @param format "HWC" or "CHW",
   *               "HWC" means (height, width, channel) order,
   *               "CHW" means (channel, height, width) order
   * @return OpenCVMat
   */
  def fromTensor(tensor: Tensor[Float], format: String = "HWC"): OpenCVMat = {
    require(format == "HWC" || format == "CHW", "the format should be HWC or CHW")
    var image = if (format == "CHW") {
      tensor.transpose(1, 2).transpose(2, 3)
    } else {
      tensor
    }
    image = image.contiguous()
    val offset = tensor.storageOffset() - 1
    var floatArr = image.storage().array()
    if (offset > 0) {
      floatArr = floatArr.slice(offset, tensor.nElement() + offset)
    }
    fromFloats(floatArr, image.size(1), image.size(2), image.size(3))
  }
}
