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

package com.intel.analytics.bigdl.nn

package object keras {
  // Alias
  val Conv1D = Convolution1D
  val Conv2D = Convolution2D
  val Conv3D = Convolution3D
  val SeparableConv2D = SeparableConvolution2D
  val AtrousConv1D = AtrousConvolution1D
  val AtrousConv2D = AtrousConvolution2D
  val Deconv2D = Deconvolution2D
}
