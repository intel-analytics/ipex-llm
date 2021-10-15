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

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.{DoubleType, FloatType, Tensor}

import scala.reflect.ClassTag

object AffineTransform3D{
  /**
   * Affine transformer implements affine transformation on a given tensor.
   * To avoid defects in resampling, the mapping is from destination to source.
   * dst(z,y,x) = src(f(z),f(y),f(x)) where f: dst -> src
   *
   * @param mat [Tensor[Double], dim: DxHxW] defines affine transformation from dst to src.
   * @param translation [Tensor[Double], dim: 3, default: (0,0,0)]
    *                    defines translation in each axis.
   * @param clampMode [String, (default: "clamp",'padding')] defines how to handle interpolation
    *                  off the input image.
   * @param padVal [Double, default: 0] defines padding value when clampMode="padding".
   *               Setting this value when clampMode="clamp" will cause an error.
   */
  def apply(mat: Tensor[Double],
            translation: Tensor[Double] = Tensor[Double](3).fill(0),
            clampMode: String = "clamp",
            padVal: Double = 0): AffineTransform3D =
      new AffineTransform3D(mat, translation, clampMode, padVal)
}

class AffineTransform3D(mat: Tensor[Double],
                        translation: Tensor[Double],
                        clampMode: String,
                        padVal: Double)
extends ImageProcessing3D {

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim >=3 && tensor.size(4) == 1,
      "Currently 3D affine transformation only supports 1 channel 3D image.")
    val src = tensor.squeeze(4)
    val dst = Tensor[Float](src.size())
    val depth = dst.size(1)
    val height = dst.size(2)
    val width = dst.size(3)
    var grid_xyz = Tensor[Double](Array[Int](3, depth, height, width))
    val cz = (depth + 1)/2.0
    val cy = (height + 1)/2.0
    val cx = (width + 1)/2.0
    for(z <- 1 to depth; y <- 1 to height; x <- 1 to width) {
      grid_xyz.setValue(1, z, y, x, cz-z)
      grid_xyz.setValue(2, z, y, x, cy-y)
      grid_xyz.setValue(3, z, y, x, cx-x)
    }
    val view_xyz = grid_xyz.reshape(Array[Int](3, depth * height * width))
    val field = mat * view_xyz
    grid_xyz = grid_xyz.sub(field.reshape(Array[Int](3, depth, height, width)))
    val translation_mat = Tensor[Double](Array[Int](3, depth, height, width))
    translation_mat(1).fill(translation.valueAt(1))
    translation_mat(2).fill(translation.valueAt(2))
    translation_mat(3).fill(translation.valueAt(3))
    grid_xyz(1) = grid_xyz(1).sub(translation_mat(1))
    grid_xyz(2) = grid_xyz(2).sub(translation_mat(2))
    grid_xyz(3) = grid_xyz(3).sub(translation_mat(3))
    val offset_mode = true
    val warp_transformer = WarpTransformer(grid_xyz, offset_mode, clampMode, padVal)
    warp_transformer(src, dst)
    dst.resize(dst.size(1), dst.size(2), dst.size(3), 1)
  }
}
