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
package com.intel.analytics.bigdl.dllib.feature.image3d

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import java.util.Calendar

import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._

object Crop3D {
  /**
   * Crop a patch from a 3D image from 'start' of patch size. The patch size should be less than
   * the image size.
   * @param start start point array(depth, height, width) for cropping
   * @param patchSize patch size array(depth, height, width)
   */
  def apply(start: Array[Int], patchSize: Array[Int]): Crop3D =
    new Crop3D(start, patchSize)

  private[bigdl] def crop(tensor: Tensor[Float], start: Array[Int],
                        patchSize: Array[Int]): Tensor[Float] = {
    require(start(0) <= tensor.size(1) && start(1) <= tensor.size(2) &&
      start(2) <= tensor.size(3), "Cropping indices out of bounds.")
    require(start(0) + patchSize(0) - 1  <= tensor.size(1)
      && start(1) + patchSize(1) - 1 <= tensor.size(2)
      && start(2) + patchSize(2) - 1 <= tensor.size(3), "Cropping indices out of bounds.")
    tensor.narrow(1, start(0), patchSize(0))
      .narrow(2, start(1), patchSize(1))
      .narrow(3, start(2), patchSize(2))
  }
}

class Crop3D(start: Array[Int], patchSize: Array[Int])
  extends ImageProcessing3D{
  require(start.size == 3 && patchSize.size == 3,
    "'start' array and 'patchSize' array should have dim 3.")
  require(patchSize(0) >= 0 && patchSize(1) >= 0 && patchSize(2) >= 0,
    "'patchSize' values should be nonnegative.")
  require(start.map(t => t >= 0).reduce((a, b) => a && b),
    "'start' values should be nonnegative.")

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    Crop3D.crop(tensor, start, patchSize)
  }
}

object RandomCrop3D {
  /**
   * Crop a random patch from an 3D image with specified patch size.
   * The patch size should be less tha the image size.
   * @param cropDepth depth after crop
   * @param cropHeight height after crop
   * @param cropWidth width after crop
   */
  def apply(cropDepth: Int, cropHeight: Int, cropWidth: Int): RandomCrop3D =
    new RandomCrop3D(cropDepth, cropHeight, cropWidth)
}

class RandomCrop3D(cropDepth: Int, cropHeight: Int, cropWidth: Int)
  extends ImageProcessing3D{

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim >= 3,
      "the transformed image array should have dim 3.")
    require(tensor.size(1) >= cropDepth,
      "the transformed image depth should be larger than cropped depth.")
    require(tensor.size(2) >= cropWidth,
      "the transformed image width should be larger than cropped width.")
    require(tensor.size(3) >= cropHeight,
      "the transformed image height should be larger than cropped height.")
    val startD = math.ceil(RNG.uniform(1e-2, tensor.size(1) - cropDepth)).toInt
    val startH = math.ceil(RNG.uniform(1e-2, tensor.size(2) - cropHeight)).toInt
    val startW = math.ceil(RNG.uniform(1e-2, tensor.size(3) - cropWidth)).toInt
    Crop3D.crop(tensor,
      Array[Int](startD, startH, startW),
      Array[Int](cropDepth, cropHeight, cropWidth))
  }
}

object CenterCrop3D {
  /**
   * Crop a `cropDepth` x `cropWidth` x `cropHeight` patch from center of image.
   * The patch size should be less than the image size.
   * @param cropDepth depth after crop
   * @param cropHeight height after crop
   * @param cropWidth width after crop
   */
  def apply(cropDepth: Int, cropHeight: Int, cropWidth: Int): CenterCrop3D =
    new CenterCrop3D(cropDepth, cropHeight, cropWidth)
}

class CenterCrop3D(cropDepth: Int, cropHeight: Int, cropWidth: Int)
  extends ImageProcessing3D{

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim >= 3,
      "the transformed image array should have dim 3.")
    require(tensor.size(1) >= cropDepth,
      "the transformed image depth should be larger than cropped depth.")
    require(tensor.size(2) >= cropHeight,
      "the transformed image width should be larger than cropped width.")
    require(tensor.size(3) >= cropWidth,
      "the transformed image height should be larger than cropped height.")
    val startD = (tensor.size(1) - cropDepth)/2
    val startH = (tensor.size(2) - cropHeight)/2
    val startW = (tensor.size(3) - cropWidth)/2
    Crop3D.crop(tensor,
      Array[Int](startD, startH, startW),
      Array[Int](cropDepth, cropHeight, cropWidth))
  }
}
