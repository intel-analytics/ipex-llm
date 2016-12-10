/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.models.imagenet

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Activities

import scala.reflect.ClassTag

import com.intel.analytics.bigdl.nn.mkl.ReLU
import com.intel.analytics.bigdl.nn.mkl.SpatialCrossMapLRN
import com.intel.analytics.bigdl.nn.mkl.Linear
import com.intel.analytics.bigdl.nn.mkl.SpatialConvolution
import com.intel.analytics.bigdl.nn.mkl.SpatialMaxPooling

import com.intel.analytics.bigdl.nn.mkl.ReLUS
import com.intel.analytics.bigdl.nn.mkl.LRN
import com.intel.analytics.bigdl.nn.mkl.Linears
import com.intel.analytics.bigdl.nn.mkl.Conv
import com.intel.analytics.bigdl.nn.mkl.MaxPooling

/**
 * @brief This is AlexNet that was presented in the One Weird Trick paper.
  *       http://arxiv.org/abs/1404.5997
 */
object AlexNet_OWT {
  def apply[T: ClassTag](classNum: Int, hasDropout : Boolean = true, firstLayerPropagateBack :
  Boolean = false)
    (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {

    val model = new Sequential[Tensor[T], Tensor[T], T]()
    model.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2).setName("conv1")
                .setNeedComputeBack(false))
    model.add(new ReLU[T](true).setName("relu1"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool1"))
    model.add(new SpatialConvolution[T](64, 192, 5, 5, 1, 1, 2, 2).setName("conv2"))
    model.add(new ReLU[T](true).setName("relu2"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool2"))
    model.add(new SpatialConvolution[T](192, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new ReLU[T](true).setName("relu3"))
    model.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1).setName("conv4"))
    model.add(new ReLU[T](true).setName("relu4"))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1).setName("conv5"))
    model.add(new ReLU[T](true).setName("relu5"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("poo5"))
    model.add(new View[T](256 * 6 * 6))
    model.add(new Linear[T](256 * 6 * 6, 4096).setName("fc6"))
    model.add(new ReLU[T](true).setName("relu6"))
    if (hasDropout) model.add(new Dropout[T](0.5).setName("drop6"))
    model.add(new Linear[T](4096, 4096).setName("fc7"))
    model.add(new ReLU[T](true).setName("relu7"))
    if (hasDropout) model.add(new Dropout[T](0.5).setName("drop7"))
    model.add(new Linear[T](4096, classNum).setName("fc8"))
    model.add(new LogSoftMax[T])
    model
  }
}

/**
 * @brief ILSVRC2012 winner
 */
object AlexNet {
  def apply[T: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val model = new Sequential[Tensor[T], Tensor[T], T]()
    model.add(new Conv[T](3, 96, 11, 11, 4, 4, propagateBack = false).setName("conv1"))
    model.add(new ReLUS[T](true).setName("relu1"))
    model.add(new LRN[T](5, 0.0001, 0.75).setName("norm1"))
    model.add(new MaxPooling[T](3, 3, 2, 2).setName("pool1"))
    model.add(new Conv[T](96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(new ReLUS[T](true).setName("relu2"))
    model.add(new LRN[T](5, 0.0001, 0.75).setName("norm2"))
    model.add(new MaxPooling[T](3, 3, 2, 2).setName("pool2"))
    model.add(new Conv[T](256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new ReLUS[T](true).setName("relu3"))
    model.add(new Conv[T](384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(new ReLUS[T](true).setName("relu4"))
    model.add(new Conv[T](384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(new ReLUS[T](true).setName("relu5"))
    model.add(new MaxPooling[T](3, 3, 2, 2).setName("pool5"))
    model.add(new View[T](256 * 6 * 6))
    model.add(new Linears[T](256 * 6 * 6, 4096).setName("fc6"))
    model.add(new ReLUS[T](true).setName("relu6"))
    model.add(new Dropout[T](0.5).setName("drop6"))
    model.add(new Linears[T](4096, 4096).setName("fc7"))
    model.add(new ReLUS[T](true).setName("relu7"))
    model.add(new Dropout[T](0.5).setName("drop7"))
    model.add(new Linears[T](4096, classNum).setName("fc8"))
    model.add(new LogSoftMax[T].setName("loss"))
    model
  }
}
