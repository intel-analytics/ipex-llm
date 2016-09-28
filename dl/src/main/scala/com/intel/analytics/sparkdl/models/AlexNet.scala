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

package com.intel.analytics.sparkdl.models

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
 */
object AlexNet_OWT {
  def apply[T: ClassTag](classNum: Int, hasDropout : Boolean = true, firstLayerPropagateBack :
  Boolean = false)
    (implicit ev: TensorNumeric[T]): Module[T] = {

    val model = new Sequential[T]
    model.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2, 1, firstLayerPropagateBack)
      .setName("conv1"))
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
 * ILSVRC2012 winner
 */
object AlexNet {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val model = new Sequential[T]()
    model.add(new SpatialConvolution[T](3, 96, 11, 11, 4, 4, 0, 0, 1, false).setName("conv1"))
    model.add(new ReLU[T](true).setName("relu1"))
    model.add(new SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm1"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool1"))
    model.add(new SpatialConvolution[T](96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(new ReLU[T](true).setName("relu2"))
    model.add(new SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm2"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool2"))
    model.add(new SpatialConvolution[T](256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new ReLU[T](true).setName("relu3"))
    model.add(new SpatialConvolution[T](384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(new ReLU[T](true).setName("relu4"))
    model.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(new ReLU[T](true).setName("relu5"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool5"))
    model.add(new View[T](256 * 6 * 6))
    model.add(new Linear[T](256 * 6 * 6, 4096).setName("fc6"))
    model.add(new ReLU[T](true).setName("relu6"))
    model.add(new Dropout[T](0.5).setName("drop6"))
    model.add(new Linear[T](4096, 4096).setName("fc7"))
    model.add(new ReLU[T](true).setName("relu7"))
    model.add(new Dropout[T](0.5).setName("drop7"))
    model.add(new Linear[T](4096, classNum).setName("fc8"))
    model.add(new LogSoftMax[T].setName("loss"))
    model
  }
}
