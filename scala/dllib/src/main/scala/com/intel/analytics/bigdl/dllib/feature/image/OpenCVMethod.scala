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

package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.opencv.OpenCV
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat

import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs

private[bigdl] object OpenCVMethod {
  OpenCV.isOpenCVLoaded

  /**
   * convert image file in bytes to opencv mat with BGR
   *
   * @param fileContent bytes from an image file
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   * @return opencv mat
   */
  def fromImageBytes(fileContent: Array[Byte],
                     imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): OpenCVMat = {
    var mat: Mat = null
    var matOfByte: MatOfByte = null
    var result: OpenCVMat = null
    try {
      matOfByte = new MatOfByte(fileContent: _*)
      mat = Imgcodecs.imdecode(matOfByte, imageCodec)
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
}
