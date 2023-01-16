/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.finetune_flickr_style

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.inception.Inception_Layer_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

object CaffeNetFlickr {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1, false)
      .setName("conv1"))
    model.add(ReLU(true).setName("relu1"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm1"))
    model.add(SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2)
      .setName("conv2"))
    model.add(ReLU(true).setName("relu2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm2"))
    model.add(SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(ReLU(true).setName("relu3"))
    model.add(SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(ReLU(true).setName("relu4"))
    model.add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(ReLU(true).setName("relu5"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool5"))
    model.add(View(256 * 6 * 6))
    model.add(Linear(256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLU(true).setName("relu6"))
    model.add(Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLU(true).setName("relu7"))
    model.add(Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, classNum).setName("fc8_flickr"))
    model.add(LogSoftMax().setName("loss"))
    model
  }
}

object Flickr_Style_Googlenet {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false).setInitMethod(Xavier)
      .setName("conv1/7x7_s2"))
    model.add(ReLU(true).setName("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3_reduce"))
    model.add(ReLU(true).setName("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3"))
    model.add(ReLU(true).setName("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75). setName("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    model.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    model.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))
    model.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    model.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    model.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))
    model.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    model.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    model.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    model.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    model.add(View(1024).setNumInputDims(3))
    model.add(Linear(1024, classNum).setInitMethod(Xavier).setName("flickr_loss3/classifier"))
    model.add(LogSoftMax().setName("loss3/loss3"))
    model.reset()
    model
  }
}


